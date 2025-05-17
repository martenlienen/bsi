#!/usr/bin/env python

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch.utilities import move_data_to_device
from omegaconf import OmegaConf
from tqdm import tqdm

from bsi.utils import print_config, set_seed
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate the ELBO")
    parser.add_argument("-s", "--split", default="test", help="Data split to evaluate")
    parser.add_argument("-c", "--checkpoint", help="Path to checkpoint", required=True)
    parser.add_argument("-o", "--out", help="Path to results json file", required=True)
    parser.add_argument(
        "-r", "--r-samples", default=1, type=int, help="Number of reconstruction samples"
    )
    parser.add_argument(
        "-m", "--m-samples", default=1, type=int, help="Number of measurement samples"
    )
    parser.add_argument(
        "-k",
        nargs="+",
        help="Number of steps for finite-step ELBO (int or 'inf')",
    )
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    split = args.split
    ckpt_path = args.checkpoint
    out_path = Path(args.out)
    r_samples = args.r_samples
    m_samples = args.m_samples
    k = args.k
    overrides = args.overrides

    try:
        k = ["inf" if s == "inf" else int(s) for s in k]
    except TypeError:
        log.error("-k takes integers or the string 'inf'")
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

    datamodule.prepare_data()
    if split == "test":
        datamodule.setup("test")
        dataloader = datamodule.test_dataloader()
    elif split == "val":
        datamodule.setup("validate")
        dataloader = datamodule.val_dataloader()
    elif split == "train":
        datamodule.setup("fit")
        dataloader = datamodule.fid_train_dataloader()
    else:
        raise RuntimeError(f"Unknown split {split}")
    if isinstance(dataloader, list):
        dataloader = dataloader[0]

    device = torch.device("cuda")
    task.to(device)
    task.eval()

    generator = torch.Generator(device).manual_seed(5410195033249451849)

    bpd_means = defaultdict(lambda: np.zeros((0,)))
    bpd_mean_vars = defaultdict(lambda: np.zeros((0,)))
    try:
        model = task.ema_algorithm
        with torch.inference_mode():
            k_bar = tqdm(k)
            for steps in k:
                k_bar.set_description(f"k = {steps}")
                batch_bar = tqdm(dataloader, desc="Batches", leave=True)
                for batch in batch_bar:
                    batch = move_data_to_device(batch, device)
                    x, _ = batch

                    if steps == "inf":
                        elbo, bpd, extra = model.elbo(
                            x, r_samples, m_samples, generator, estimate_var=True
                        )
                    else:
                        if isinstance(model, VDM):
                            t = torch.linspace(1.0, 0.0, steps + 1, device=device)
                        else:
                            t = torch.linspace(0.0, 1.0, steps + 1, device=device)

                        elbo, bpd, extra = model.finite_elbo(
                            x, r_samples, m_samples, generator, estimate_var=True, t=t
                        )

                    bpd_means[steps] = np.concat(
                        (bpd_means[steps], bpd.numpy(force=True))
                    )
                    bpd_mean_vars[steps] = np.concat(
                        (bpd_mean_vars[steps], extra["bpd_var"].numpy(force=True))
                    )

                    mc_std = np.sqrt(
                        (bpd_means[steps].var(ddof=1) + bpd_mean_vars[steps].mean())
                        / len(bpd_means[steps])
                    )
                    batch_bar.set_postfix(
                        {"bpd": f"{bpd_means[steps].mean().item():.4f} +- {mc_std:.4f}"}
                    )
                batch_bar.close()

                n = len(bpd_means[steps])
                steps_mean = bpd_means[steps].mean()
                steps_mean_var = (
                    bpd_means[steps].var(ddof=1) + bpd_mean_vars[steps].mean()
                ) / n
                bpd_means[steps] = steps_mean
                bpd_mean_vars[steps] = steps_mean_var
    finally:
        results = {
            "ckpt": str(ckpt_path),
            "config": {
                "split": split,
                "r_samples": r_samples,
                "m_samples": m_samples,
                "k": k,
                "overrides": overrides,
            },
            "bpd_means": {k: means.tolist() for k, means in bpd_means.items()},
            "bpd_mean_vars": {k: vars.tolist() for k, vars in bpd_mean_vars.items()},
        }

        out_path.parent.mkdir(exist_ok=True, parents=True)
        out_path.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
