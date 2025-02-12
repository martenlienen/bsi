#!/usr/bin/env python

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Evaluate the ELBO")
    parser.add_argument("-s", "--split", default="test", help="Data split to evaluate")
    parser.add_argument("-c", "--checkpoint", help="Path to checkpoint", required=True)
    parser.add_argument("-o", "--out", help="Path to results pickle file", required=True)
    parser.add_argument(
        "-r", "--r-samples", default=1, type=int, help="Number of reconstruction samples"
    )
    parser.add_argument(
        "-m", "--m-samples", default=1, type=int, help="Number of measurement samples"
    )
    parser.add_argument(
        "-k", default=None, type=int, help="Number of steps for finite-step ELBO"
    )
    parser.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    split = args.split
    ckpt_path = args.checkpoint
    out_path = args.out
    r_samples = args.r_samples
    m_samples = args.m_samples
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

    datamodule.prepare_data()
    if split == "test":
        datamodule.setup("test")
        dataloader = datamodule.test_dataloader()
    elif split == "val":
        datamodule.setup("validate")
        dataloader = datamodule.val_dataloader()
    elif split == "train":
        datamodule.setup("fit")
        dataloader = datamodule.train_dataloader()
    else:
        raise RuntimeError(f"Unknown split {split}")
    if isinstance(dataloader, list):
        dataloader = dataloader[0]

    device = torch.device("cuda")
    task.to(device)
    task.eval()

    generator = torch.Generator(device).manual_seed(5410195033249451849)

    bpds = []
    l_recons = []
    l_measures = []
    try:
        bsi = task.ema_bsi
        with torch.inference_mode():
            bar = tqdm(dataloader)
            for batch in bar:
                batch = move_data_to_device(batch, device)
                x, _ = batch

                n_dim = math.prod(bsi.data_shape)
                l_recon = bsi.reconstruction_loss(x, r_samples, generator)
                if k is None:
                    l_measure = bsi.inf_measurement_loss(
                        x, m_samples, generator, return_samples=True
                    )
                else:
                    t = torch.linspace(0.0, 1.0, k, device=device)
                    l_measure = bsi.finite_measurement_loss(
                        x, m_samples, generator, t=t, return_samples=True
                    )

                # Bits per dimension
                bpd = (l_recon + l_measure) / (math.log(2) * n_dim)

                bpds.append(bpd.cpu())
                l_recons.append(l_recon.cpu())
                l_measures.append(l_measure.cpu())

                all_bpds = torch.cat(bpds)
                bar.set_postfix(
                    {
                        "bpd": f"{all_bpds.mean().item():.4f}",
                        "std": np.sqrt(all_bpds.var().item() / len(all_bpds)),
                    }
                )
    finally:
        bpds = torch.cat(bpds).flatten().numpy()
        l_recons = torch.cat(l_recons).flatten().numpy()
        l_measures = torch.cat(l_measures).flatten().numpy()
        print(
            f"bpd: {bpds.mean()}, recon: {l_recons.mean()}, measure: {l_measures.mean()}"
        )
        meta = {
            "ckpt": str(ckpt_path),
            "config": {
                "split": split,
                "r_samples": r_samples,
                "m_samples": m_samples,
                "k": k,
                "overrides": overrides,
            },
        }
        full_results = {"bpd": bpds, "l_recon": l_recons, "l_measure": l_measures, **meta}

        out_root = (
            Path(out_path)
            / datamodule.short_name()
            / f"{split}"
            / config.logging.wandb.id
        )
        out_root.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(
            out_root / f"elbo-{k}-r{r_samples}-m{m_samples}.npz", **full_results
        )

        summarized_results = {
            "bpd": float(bpds.mean()),
            "l_recon": float(l_recons.mean()),
            "l_measure": float(l_measures.mean()),
            **meta,
        }
        (out_root / f"elbo-{k}-r{r_samples}-m{m_samples}.json").write_text(
            json.dumps(summarized_results)
        )


if __name__ == "__main__":
    main()
