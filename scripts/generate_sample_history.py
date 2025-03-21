#!/usr/bin/env python

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bsi.utils import print_config, set_seed

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


def to_image(x, discretization):
    x = discretization.to_unit_interval(x)
    return (255 * x.clamp(min=0.0, max=1.0)).to(torch.uint8)


def main():
    parser = argparse.ArgumentParser(description="Generate samples from a model")
    parser.add_argument("-c", "--checkpoint", help="Path to checkpoint", required=True)
    parser.add_argument("-o", "--out", help="Path to output file", required=True)
    parser.add_argument("-n", type=int, default=256, help="Number of samples to generate")
    parser.add_argument("-k", type=int, help="Number of sample steps", required=True)
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    ckpt_path = args.checkpoint
    out_path = Path(args.out)
    n = args.n
    k = args.k
    overrides = args.overrides

    ckpt = torch.load(ckpt_path)
    if "config" in ckpt:
        log.info("Load config from checkpoint")
        run_config = OmegaConf.create(ckpt["config"])
    else:
        log.error("Checkpoint has no config")
        sys.exit(1)
    config = OmegaConf.merge(run_config, OmegaConf.from_cli(overrides))
    rng, seed_sequence = set_seed(config)

    print_config(config)

    torch.set_float32_matmul_precision(config.matmul_precision)

    datamodule = instantiate(config.data)
    task = instantiate(config.task, datamodule=datamodule, seed_sequence=seed_sequence)
    task.load_state_dict(ckpt["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task.to(device)
    task.eval()

    generator = torch.Generator(device).manual_seed(16213294677523980332)
    with torch.inference_mode():
        t = torch.linspace(0.0, 1.0, k + 1, device=device)
        lambdas = task.ema_bsi.p_lambda.icdf(t)
        mus, x_hats, ys = task.ema_bsi.sample_history(n, generator=generator, t=t)
        mus = to_image(mus, task.discretization)
        x_hats = to_image(x_hats, task.discretization)
        ys = to_image(ys, task.discretization)
    np.savez(
        out_path,
        lambdas=lambdas.numpy(force=True),
        mus=mus.numpy(force=True),
        x_hats=x_hats.numpy(force=True),
        ys=ys.numpy(force=True),
    )


if __name__ == "__main__":
    main()
