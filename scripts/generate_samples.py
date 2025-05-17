#!/usr/bin/env python

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from tqdm import tqdm

from bsi.tasks.vdm import VDMTraining
from bsi.utils import print_config, relative_to_project_root, set_seed

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


def get_batch_sizes(num_samples: int, batch_size: int) -> List[int]:
    batch_sizes = [batch_size] * (num_samples // batch_size)

    if (remaining_samples := num_samples % batch_size) > 0:
        batch_sizes = batch_sizes + [remaining_samples]

    return batch_sizes


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


def main():
    parser = argparse.ArgumentParser(description="Generate samples from a model")
    parser.add_argument("-c", "--checkpoint", help="Path to checkpoint", required=True)
    parser.add_argument("-o", "--out", help="Path to output folder", required=True)
    parser.add_argument("-n", "--num-samples", type=int, help="Num samples for FID")
    parser.add_argument("-s", "--schedule", default="linear", help="Schedule name")
    parser.add_argument(
        "-e", "--noema", action=argparse.BooleanOptionalAction, help="Disable EMA"
    )
    parser.add_argument("-k", type=int, help="Number of sample steps", required=True)
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    ckpt_path = args.checkpoint
    out_path = Path(args.out)
    num_samples = args.num_samples
    schedule_name = args.schedule
    no_ema = args.noema
    k = args.k
    overrides = args.overrides

    if out_path.exists() and not out_path.is_dir():
        log.error(f"{out_path} exists and is not a directory")
        sys.exit()

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

    if num_samples is None:
        num_samples = fid_stats(datamodule.short_name(), "test")["n"].item()

    generator = torch.Generator(device).manual_seed(16213294677523980332)
    if hasattr(task, "bsi"):
        max_variance = 1 / task.bsi.lambda_0
        min_variance = 1 / (task.bsi.lambda_0 + task.bsi.alpha_M)
    match schedule_name:
        case "linear":
            if isinstance(task, VDMTraining):
                t = torch.linspace(1, 0, k + 1, device=device)
            else:
                t = torch.linspace(0, 1, k + 1, device=device)
        case "cosine":
            # Compute variances of cosine schedule here going from largest to smallest
            variance = (max_variance - min_variance) * torch.cos(
                torch.linspace(0, 1, k + 1, device=device) * torch.pi / 2
            ) ** 2 + min_variance
            t = task.bsi.p_lambda.cdf(1 / variance)
        case "edm":
            variance = (
                torch.linspace(
                    max_variance.sqrt(), min_variance.sqrt(), k + 1, device=device
                )
                ** 2
            )
            t = task.bsi.p_lambda.cdf(1 / variance)
        case "edm7":
            t = torch.linspace(0, 1, k + 1, device=device)
            max_std = max_variance.sqrt()
            min_std = min_variance.sqrt()
            rho = 7
            stds = (
                max_std ** (1 / rho) + t * (min_std ** (1 / rho) - max_std ** (1 / rho))
            ) ** (rho)
            variance = stds**2
            t = task.bsi.p_lambda.cdf(1 / variance)

        case _:
            log.error(f"Unknown schedule {schedule_name}")

    try:
        batch_sizes = get_batch_sizes(
            num_samples=num_samples,
            batch_size=config.data.eval_batch_size,
        )

        algorithm = task.algorithm if no_ema else task.ema_algorithm

        samples = []
        fid_embeddings = []
        with torch.inference_mode():
            bar = tqdm(desc="Samples", total=num_samples)
            for batch_size in batch_sizes:
                batch = algorithm.sample(batch_size, generator=generator, t=t)
                batch = task.discretization.to_unit_interval(batch)

                samples.append(batch.cpu())
                images = (255 * batch.clamp(min=0.0, max=1.0)).to(torch.uint8)
                fid_embeddings.append(fid.inception(images))

                bar.update(batch_size)
    finally:
        samples = torch.cat(samples, dim=0)
        fid_embeddings = torch.cat(fid_embeddings, dim=0)

        out_root = (
            out_path
            / datamodule.short_name()
            / (config.logging.wandb.get("id") or "unknown")
            / f"{schedule_name}-{k}"
        )
        out_root.mkdir(parents=True, exist_ok=True)
        meta = {"ckpt": str(ckpt_path), "config": {"overrides": overrides}}
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")

        sample_results = {"samples": samples, "fid_embeddings": fid_embeddings, **meta}
        torch.save(sample_results, out_root / f"samples-{num_samples}-{timestamp}.pt")

        embs = fid_embeddings.double()
        mean = embs.mean(dim=0).cpu()
        cov = embs.T.cov().cpu()

        fid = {
            stage: compute_fid(mean, cov, datamodule.short_name(), stage)
            for stage in ["train", "test"]
        }
        print("FID:", fid)

        fid_results = {"fid": fid, **meta}
        torch.save(fid_results, out_root / f"fid-{num_samples}-{timestamp}.pt")


if __name__ == "__main__":
    main()
