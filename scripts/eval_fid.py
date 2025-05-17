#!/usr/bin/env python

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from tqdm import tqdm

from bsi.utils import print_config, relative_to_project_root, set_seed
from bsi.vdm import VDM

warnings.filterwarnings(
    "ignore",
    "The '\\w+_dataloader' does not have many workers",
    module="lightning",
)
warnings.filterwarnings(
    "ignore",
    "The `srun` command is available on your system but is not used",
    module="lightning",
)

log = logging.getLogger(__name__)


def fid_stats(dataset_name: str, stage: str):
    stats_path = relative_to_project_root(
        Path() / "data" / "fid-stats" / dataset_name / f"{stage}.npz"
    )
    return np.load(stats_path)


def compute_fid(mean, cov, dataset_name: str, stage: str) -> float:
    stats = fid_stats(dataset_name, stage)
    data_n = stats["n"].item()
    data_sum = torch.as_tensor(stats["sum"], dtype=torch.double)
    data_cov_sum = torch.as_tensor(stats["cov_sum"], dtype=torch.double)
    data_mean = data_sum / data_n
    data_cov = (data_cov_sum - data_n * torch.outer(data_mean, data_mean)) / (data_n - 1)

    return _compute_fid(mean, cov, data_mean, data_cov).item()


def get_schedule(name: str, k: int, device: torch.device, model) -> torch.Tensor:
    if name == "linear":
        if isinstance(model, VDM):
            return torch.linspace(1, 0, k + 1, device=device)
        else:
            return torch.linspace(0, 1, k + 1, device=device)

    # TODO: Flip these for VDM
    max_variance = 1 / model.lambda_0
    min_variance = 1 / (model.lambda_0 + model.alpha_M)
    match name:
        case "cosine":
            # Compute variances of cosine schedule here going from largest to smallest
            variance = (max_variance - min_variance) * torch.cos(
                torch.linspace(0, 1, k, device=device) * torch.pi / 2
            ) ** 2 + min_variance
            return model.p_lambda.cdf(1 / variance)
        case "edm":
            variance = (
                torch.linspace(max_variance.sqrt(), min_variance.sqrt(), k, device=device)
                ** 2
            )
            t = model.p_lambda.cdf(1 / variance)
        case "edm7":
            t = torch.linspace(0, 1, k, device=device)
            max_std = max_variance.sqrt()
            min_std = min_variance.sqrt()
            rho = 7
            stds = (
                max_std ** (1 / rho) + t * (min_std ** (1 / rho) - max_std ** (1 / rho))
            ) ** (rho)
            variance = stds**2
            return model.p_lambda.cdf(1 / variance)
        case _:
            log.error(f"Unknown schedule {name}")
            sys.exit()


def get_batch_sizes(n: int, batch_size: int) -> list[int]:
    batch_sizes = [batch_size] * (n // batch_size)

    if (remaining_samples := n % batch_size) > 0:
        batch_sizes = batch_sizes + [remaining_samples]

    return batch_sizes


def main():
    parser = argparse.ArgumentParser(description="Evaluate the FID")
    parser.add_argument("-c", "--checkpoint", help="Path to checkpoint", required=True)
    parser.add_argument("-o", "--out", help="Path to results json file", required=True)
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples for FID")
    parser.add_argument("-s", "--schedule", default="linear", help="Schedule name")
    parser.add_argument(
        "-k", nargs="+", type=int, help="Number of sample steps", required=True
    )
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    ckpt_path = args.checkpoint
    out_path = Path(args.out)
    n = args.num_samples
    schedule_name = args.schedule
    ks = args.k
    overrides = args.overrides

    ckpt = torch.load(ckpt_path, weights_only=False)
    if "config" in ckpt:
        log.info("Load config from checkpoint")
        run_config = OmegaConf.create(ckpt["config"])
    else:
        log.error("Checkpoint has no config")
        sys.exit()
    config = OmegaConf.merge(run_config, OmegaConf.from_cli(overrides))
    rng, seed_sequence = set_seed(config)

    print_config(config)

    torch.set_float32_matmul_precision(config.matmul_precision)

    datamodule = instantiate(config.data)
    task = instantiate(config.task, datamodule=datamodule, seed_sequence=seed_sequence)
    task.load_state_dict(ckpt["state_dict"])

    device = torch.device("cuda")
    task.to(device)
    task.eval()

    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    fid.to(device)

    if n is None:
        n = fid_stats(datamodule.short_name(), "test")["n"].item()

    generator = torch.Generator(device).manual_seed(5410195033249451849)

    try:
        algorithm = task.ema_algorithm
        with torch.inference_mode():
            fids = {}
            k_bar = tqdm(ks)
            for k in ks:
                k_bar.set_description(f"k = {k}")
                t = get_schedule(schedule_name, k, device, algorithm)
                batch_bar = tqdm(
                    desc="Samples",
                    leave=True,
                    total=n,
                )
                fid_embeddings = []
                for batch_size in get_batch_sizes(n, config.data.eval_batch_size):
                    batch = algorithm.sample(batch_size, generator=generator, t=t)
                    batch = task.discretization.to_unit_interval(batch)

                    images = (255 * batch.clamp(min=0.0, max=1.0)).to(torch.uint8)
                    fid_embeddings.append(fid.inception(images))

                    batch_bar.update(batch_size)

                embs = torch.cat(fid_embeddings, dim=0).double()
                mean = embs.mean(dim=0).cpu()
                cov = embs.T.cov().cpu()
                fids[k] = {
                    stage: compute_fid(mean, cov, datamodule.short_name(), stage)
                    for stage in ["train", "test"]
                }

                batch_bar.set_postfix(fids[k])
                batch_bar.close()
    finally:
        results = {
            "ckpt": str(ckpt_path),
            "config": {
                "n": n,
                "k": k,
                "schedule": schedule_name,
                "overrides": overrides,
            },
            "fid": fids,
        }

        out_path.parent.mkdir(exist_ok=True, parents=True)
        out_path.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
