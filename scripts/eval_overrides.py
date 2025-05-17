#!/usr/bin/env python

import argparse
import logging
import sys
import warnings

import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from bsi.lightning.plugins import TrainOnlyAMP
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


def get_plugins(config):
    plugins = []
    if config.trainer.precision in ("16-mixed", "bf16-mixed"):
        amp_plugin = TrainOnlyAMP(config.trainer.precision)
        with open_dict(config):
            del config.trainer.precision
        plugins.append(amp_plugin)
    return plugins


def main():
    parser = argparse.ArgumentParser(description="Re-test a checkpoint with overrides")
    parser.add_argument("-s", "--split", default="test", help="Data split to evaluate")
    parser.add_argument("-c", "--checkpoint", help="Path to checkpoint")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    split = args.split
    ckpt_path = args.checkpoint
    overrides = args.overrides

    ckpt = torch.load(ckpt_path, weights_only=False) if ckpt_path is not None else {}
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

    trainer = Trainer(**config.trainer, plugins=get_plugins(config), logger=False)

    datamodule.prepare_data()
    if split == "test":
        datamodule.setup("test")
        dataloaders = datamodule.test_dataloader()
    elif split == "val":
        datamodule.setup("validate")
        dataloaders = datamodule.val_dataloader()
    else:
        logger.error(f"Unknown data split {split}")
        sys.exit()

    metrics = trainer.test(model=task, ckpt_path=ckpt_path, dataloaders=dataloaders)
    if not isinstance(metrics, list):
        metrics = [metrics]

    suffix = "-".join(overrides)
    for dataloader_metrics in metrics:
        suffixed_metrics = {f"{k}-{suffix}": v for k, v in dataloader_metrics.items()}

        print(suffixed_metrics)


if __name__ == "__main__":
    main()
