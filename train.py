#!/usr/bin/env python

import faulthandler
import logging
import os
import socket
import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from omegaconf import DictConfig, OmegaConf, open_dict
from submitit import JobEnvironment

from bsi.lightning.callbacks import ConfigInCheckpoint
from bsi.lightning.plugins import TrainOnlyAMP
from bsi.lightning.strategies import NoWrappingDDPStrategy
from bsi.utils import (
    filter_device_available,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)

# Let's us do arithmetic (and anything else) in configurations
OmegaConf.register_new_resolver("eval", eval)


log = logging.getLogger("bsi")


def global_setup():
    # There is some weird interaction where /tmp is no longer writable when slurm sends
    # the process a SIGUSR1 because of a job timeout. As a consequence, lightning cannot
    # store its checkpoint (which uses /tmp for technical reasons) and thus requeuing
    # and the whole training fail.
    tmpdir = Path("~/tmp").expanduser()
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)

    # Log to traceback to stderr on segfault
    faulthandler.enable(all_threads=False)

    # If data loading is really not a bottleneck for you, uncomment this to silence the
    # warning about it
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
    logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
        filter_device_available
    )


global_setup()


def store_job_info(config: DictConfig):
    host = socket.gethostname()
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    process_id = os.getpid()

    with open_dict(config):
        config.host = host
        config.process_id = str(process_id)
        if array_job_id is not None and array_task_id is not None:
            config.slurm_job_id = f"{array_job_id}_{array_task_id}"
        elif job_id is not None:
            config.slurm_job_id = job_id


def restore_resumed_wandb_run_id(config: DictConfig):
    """If this is requeued/resumed run, restore the w&b run ID in the config."""

    if "SUBMITIT_FOLDER" not in os.environ:
        return
    run_id_file = JobEnvironment().paths.folder / "wandb_run_id"
    if not run_id_file.is_file():
        return

    with open_dict(config):
        config.logging.wandb.id = run_id_file.read_text()


def store_wandb_run_id(run_id: str):
    """Store the w&b run ID on disk, so that we can resume the run after a timeout."""

    if "SUBMITIT_FOLDER" not in os.environ:
        return

    (JobEnvironment().paths.folder / "wandb_run_id").write_text(run_id)


def get_callbacks(config, logger):
    callbacks = [
        ModelCheckpoint(
            dirpath=f"runs/{logger.name}/{logger.version}",
            save_top_k=-1,
            # This ensures that a checkpoint is saved after every validation
            every_n_epochs=1,
        ),
        TQDMProgressBar(refresh_rate=1),
        LearningRateMonitor(),
        ConfigInCheckpoint(config),
    ]
    return callbacks


def get_plugins(config):
    plugins = []
    if config.trainer.precision in ("16-mixed", "bf16-mixed"):
        amp_plugin = TrainOnlyAMP(config.trainer.precision)
        with open_dict(config):
            del config.trainer.precision
        plugins.append(amp_plugin)
    return plugins


def get_strategy(config):
    if NoWrappingDDPStrategy.applicable():
        return NoWrappingDDPStrategy(start_method="popen")
    return "auto"


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    # Global setup is inside main, so that it also runs on the compute node when we submit
    # the jobs with submitit. Otherwise, some module level settings such as warning
    # filters would be lost when the modules are re-imported fresh on the compute node.
    global_setup()

    ##################
    # Model Training #
    ##################

    if NoWrappingDDPStrategy.applicable() and config.seed is None:
        log.error("You need to fix a seed when running multi-GPU!")
        raise SystemExit(1)

    rng, seed_sequence = set_seed(config)

    # Log host and slurm job ID
    store_job_info(config)

    # Log to same W&B ID if this is a requeued run after a timeout
    restore_resumed_wandb_run_id(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    print_config(config)

    torch.set_float32_matmul_precision(config.matmul_precision)

    log.info("Loading data")
    datamodule = instantiate(config.data)

    log.info("Instantiating model")
    task = instantiate(
        config.task, datamodule=datamodule, seed_sequence=seed_sequence.spawn(1)[0]
    )

    logger = instantiate(
        config.logging.wandb,
        _target_="lightning.pytorch.loggers.WandbLogger",
        resume=(config.logging.wandb.mode == "online") and "allow",
        # Don't upload any checkpoints to save space
        log_model=False,
        # Use trainer's default_root_dir
        save_dir=None,
    )
    # Ensure that wandb is initialized
    logger.experiment

    # Store the W&B run ID on disk so that we can resume the run after a timeout
    if logger.version is not None:
        store_wandb_run_id(logger.version)

    log_hyperparameters(logger, config, task)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config, logger)
    plugins = get_plugins(config)
    strategy = get_strategy(config)
    trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        plugins=plugins,
        strategy=strategy,
        logger=logger,
        default_root_dir=f"runs/{logger.name}/{logger.version}",
    )

    fit_kwargs = {}
    if config.from_ckpt is not None:
        fit_kwargs["ckpt_path"] = config.from_ckpt

    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule, **fit_kwargs)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    logger.finalize("success")
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
