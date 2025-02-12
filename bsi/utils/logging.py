import re

import lightning.pytorch as pl
import rich
import torch.nn as nn
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.syntax import Syntax

DEVICE_AVAILABLE = re.compile("[TIH]PU available: ")


def filter_device_available(record):
    """Filter the availability report for all the devices we don't have."""
    return not DEVICE_AVAILABLE.match(record.msg)


@rank_zero_only
def print_config(config: DictConfig) -> None:
    content = OmegaConf.to_yaml(config, resolve=True)
    rich.print(Syntax(content, "yaml"))


def count_params(model: nn.Module):
    return {
        "params-total": sum(p.numel() for p in model.parameters()),
        "params-trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "params-not-trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
    }


@rank_zero_only
def log_hyperparameters(
    logger: pl.loggers.Logger, config: DictConfig, model: pl.LightningModule
):
    hparams = OmegaConf.to_container(config, resolve=True)
    hparams.setdefault("model", {}).update(count_params(model))

    logger.log_hyperparams(hparams)

    # Disable logging any more hyperparameters for all loggers (this is just a trick to
    # prevent trainer from logging hparams of model, since we already did that above)
    logger.log_hyperparams = lambda *args, **kwargs: None
