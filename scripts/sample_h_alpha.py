#!/usr/bin/env python

import argparse
import logging
import math
import sys
import warnings

import einops as eo
import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch.utilities import move_data_to_device
from omegaconf import OmegaConf
from tqdm import tqdm

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


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--checkpoint", help="Path to checkpoint", required=True)
    parser.add_argument("-n", default=1000, type=int, help="Number of lambda steps")
    parser.add_argument("-o", "--out", help="Path to results pickle file", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    n = args.n
    out_path = args.out
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

    datamodule.prepare_data()
    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()
    if isinstance(dataloader, list):
        dataloader = dataloader[0]

    device = torch.device("cuda")
    task.to(device)
    task.eval()

    model = task.ema_bsi
    generator = torch.Generator(device).manual_seed(2363185049904024905)
    lambdas = torch.logspace(
        torch.log10(model.lambda_0),
        torch.log10(model.lambda_0 + model.alpha_M),
        n,
        device=device,
    )
    t = model.p_lambda.cdf(lambdas)

    errors_bpd = []
    try:
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Batches"):
                batch = move_data_to_device(batch, device)
                x, _ = batch

                batch_size = len(x)
                mu = model._sample_q_mu_lambda(
                    x, eo.repeat(lambdas, "n -> n b", b=batch_size), generator
                )
                x_hat = model._predict_x(
                    mu.flatten(end_dim=1), eo.repeat(t, "n -> (n b)", b=batch_size)
                )
                x_hat = eo.rearrange(x_hat, "(n b) ... -> n b ...", n=n)
                decoding_error = eo.reduce(
                    (x - x_hat).square(), "n batch ... -> n batch", "mean"
                )

                errors_bpd.append((decoding_error / math.log(2)).cpu())
    finally:
        errors_bpd = torch.cat(errors_bpd, dim=1)

        results = {
            "ckpt": str(ckpt_path),
            "lambda": lambdas.numpy(force=True),
            "squared_error_samples_bpd": errors_bpd.numpy(force=True),
        }
        np.savez_compressed(out_path, **results)


if __name__ == "__main__":
    main()
