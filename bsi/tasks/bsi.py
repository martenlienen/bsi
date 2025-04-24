import logging
from pathlib import Path

import einops as eo
import lightning.pytorch as pl
import numpy as np
import torch
import torchmetrics as tm
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel

from ..bsi import BSI, Discretization
from ..utils import torch_generator_seed
from ..utils.path import relative_to_project_root
from .ema_pytorch import EMA
from .metrics.fid import FIDScore

log = logging.getLogger(__name__)


class Plots(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module: "BSITraining"):
        plots = {}

        plot_generator = torch.Generator(pl_module.device).manual_seed(2831183658)
        discretization = pl_module.discretization

        samples = pl_module._sample(64, plot_generator)
        assert torch.all(torch.isfinite(samples))
        image = eo.rearrange(
            discretization.to_8bit_image(samples), "(a b) c h w -> (b h) (a w) c", a=8
        )
        plots["val/samples"] = wandb.Image(image.numpy(force=True))

        _, x_hats, _ = pl_module._sample_history(16, plot_generator)
        assert torch.all(torch.isfinite(x_hats))
        histories_img = eo.rearrange(
            discretization.to_8bit_image(x_hats),
            "hist batch c h w -> (batch h) (hist w) c",
        )
        plots["val/histories"] = wandb.Image(histories_img.numpy(force=True))

        tensor_args = {"dtype": pl_module.dtype, "device": pl_module.device}
        quantiles = torch.linspace(0.0, 1.0, 15, **tensor_args)
        data, _ = trainer.datamodule.train_data[
            np.linspace(
                0, min(len(trainer.datamodule.train_data) - 1, 1000), num=8, dtype=int
            ).tolist()
        ]
        data = data.to(**tensor_args)
        lambda_ = eo.repeat(
            pl_module.bsi.p_lambda.icdf(quantiles), "i -> i b", b=len(data)
        )
        mu = pl_module.bsi._sample_q_mu_lambda(data, lambda_, plot_generator).flatten(
            end_dim=1
        )
        x_hat = pl_module.bsi._predict_x(
            mu, eo.repeat(quantiles, "i -> (i b)", b=len(data))
        )
        assert torch.all(torch.isfinite(x_hat))
        denoisings = eo.rearrange(
            discretization.to_8bit_image(torch.stack((mu, x_hat))),
            "stack (alphas batch) c h w -> (batch stack h) (alphas w) c",
            batch=len(data),
        )
        plots["val/denoisings"] = wandb.Image(denoisings.numpy(force=True))

        trainer.logger.log_metrics(plots, step=trainer.global_step)


def create_ema(model, beta=0.9999, update_after_step=100, update_every=10, **kwargs):
    return EMA(
        model,
        beta=beta,
        update_after_step=update_after_step,
        update_every=update_every,
        include_online_model=False,
        use_foreach=True,
    )


class BSITraining(pl.LightningModule):
    def __init__(
        self,
        datamodule,
        bsi: DictConfig,
        model: DictConfig,
        ema: DictConfig | None,
        compile: bool,
        compile_mode: str | None,
        n_elbo_recon_samples: int,
        n_elbo_measure_samples: int,
        optimizer: DictConfig,
        seed_sequence: np.random.SeedSequence,
        lr_scheduler: DictConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("datamodule",))

        data_shape = datamodule.data_shape()
        self.discretization = Discretization.image_8bit()
        self.model = instantiate(model, data_shape=data_shape)
        bsi_kwargs = dict(data_shape=data_shape, discretization=self.discretization)
        self.bsi: BSI = instantiate(
            bsi, model=self.model, **bsi_kwargs, _convert_="object"
        )

        if ema is None:
            self.ema_model = None
            self.ema_bsi = None
        else:
            self.ema_model = create_ema(self.model, **ema)
            self.ema_bsi: BSI = instantiate(
                bsi, model=self.ema_model, **bsi_kwargs, _convert_="object"
            )

        # Compile individual BSI methods
        self._train_loss = self.bsi.train_loss
        if self.ema_bsi is None:
            self._elbo = self.bsi.elbo
            self._sample = self.bsi.sample
            self._sample_history = self.bsi.sample_history
        else:
            self._elbo = self.ema_bsi.elbo
            self._sample = self.ema_bsi.sample
            self._sample_history = self.ema_bsi.sample_history
        if compile:
            self._train_loss = torch.compile(self._train_loss, mode=compile_mode)
            self._elbo = torch.compile(self._elbo, mode=compile_mode)
            self._sample = torch.compile(self._sample, mode=compile_mode)
            self._sample_history = torch.compile(self._sample_history, mode=compile_mode)

        self.n_elbo_recon_samples = n_elbo_recon_samples
        self.n_elbo_measure_samples = n_elbo_measure_samples

        self.seed_sequence = seed_sequence
        self.train_generator_seed = torch_generator_seed(seed_sequence)
        self.val_generator_seed = torch_generator_seed(seed_sequence)
        self.test_generator_seed = torch_generator_seed(seed_sequence)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.val_sample_metrics = self._metrics("val", datamodule)
        self.test_sample_metrics = self._metrics("test", datamodule)

        # On some datasets, e.g. CIFAR10, the sample metrics are commonly evaluated
        # against the train set, so we also compute them for the train set.
        self.any_train_samples = False
        self.train_sample_metrics = self._metrics("train", datamodule)

    def configure_ddp(self):
        # Compilation applies lazily, so we can just update it with the wrapped model
        self.bsi.set_model(DistributedDataParallel(self.model, static_graph=True))
        assert isinstance(self.bsi.model, DistributedDataParallel)

    def _metrics(self, stage: str, datamodule):
        sample_metrics = {}
        if datamodule.data_shape()[0] == 3:
            stats_path = relative_to_project_root(
                Path() / "data" / "fid-stats" / datamodule.short_name() / f"{stage}.npz"
            )
            if stats_path.is_file():
                sample_metrics["fid-2048"] = FIDScore(feature=2048, stats_path=stats_path)
            else:
                log.warning(f"No precomputed FID statistics for {stage} found.")
        return tm.MetricCollection(sample_metrics, prefix=f"{stage}/")

    def _generator(self, seed: int):
        # Initialize generators lazily to ensure that they are on the correct device
        return torch.Generator(self.device).manual_seed(seed)

    def on_train_start(self):
        self.train_generator = self._generator(self.train_generator_seed)

    def training_step(self, batch, batch_idx):
        data, _ = batch
        data = data.to(dtype=self.dtype)

        loss = self._train_loss(data, self.train_generator).mean()

        self.log("train/loss", loss, batch_size=len(data))
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_model is not None:
            self.ema_model.update()

    def on_validation_epoch_start(self):
        # Reset the validation generator to reduce validation variance between runs
        self.val_generator = self._generator(self.val_generator_seed)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            stage = "val"
            metrics = self.val_sample_metrics
        elif dataloader_idx == 1:
            self.any_train_samples = True
            stage = "train"
            metrics = self.train_sample_metrics
        else:
            log.warning(f"Unknown data loader index {dataloader_idx}")
            return

        return self.eval_step(stage, batch, batch_idx, metrics, self.val_generator)

    def eval_step(
        self, stage: str, batch, batch_idx, sample_metrics, generator: torch.Generator
    ):
        data, _ = batch
        data = data.to(dtype=self.dtype)
        batch_size = len(data)

        # Reduce number of samples during sanity checking to make it fast
        if self.trainer.sanity_checking:
            batch_size = min(batch_size, 16)
            data = data[:batch_size]

        elbo, bpd, l_recon, l_measure = self._elbo(
            data,
            n_recon_samples=self.n_elbo_recon_samples,
            n_measure_samples=self.n_elbo_measure_samples,
            generator=generator,
        )
        elbo_metrics = {
            f"{stage}/elbo": elbo.mean(),
            f"{stage}/bpd": bpd.mean(),
            f"{stage}/l_recon": l_recon.mean(),
            f"{stage}/l_measure": l_measure.mean(),
        }
        self.log_dict(elbo_metrics, batch_size=batch_size)

        if len(sample_metrics) > 0:
            samples = self._sample(batch_size, generator)
            # Clamp to avoid integer under or overflow during uint8 conversion in FID
            samples = self.discretization.to_unit_interval(samples).clamp(
                min=0.0, max=1.0
            )
            sample_metrics.update(samples)
            return {"samples": samples}
        else:
            return {}

    def on_validation_epoch_end(self):
        self.log_dict(self.val_sample_metrics.compute())
        self.val_sample_metrics.reset()

        if self.any_train_samples:
            self.log_dict(self.train_sample_metrics.compute())
            self.train_sample_metrics.reset()
        else:
            log.info(
                "No train samples generated at validation time, so no metrics to log."
            )

    def on_test_epoch_start(self):
        # Reset the test generator to reduce test variance between runs
        self.test_generator = self._generator(self.test_generator_seed)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            stage = "test"
            metrics = self.test_sample_metrics
        elif dataloader_idx == 1:
            self.any_train_samples = True
            stage = "train"
            metrics = self.train_sample_metrics
        else:
            log.warning(f"Unknown data loader index {dataloader_idx}")
            return

        return self.eval_step(stage, batch, batch_idx, metrics, self.test_generator)

    def on_test_epoch_end(self):
        self.log_dict(self.test_sample_metrics.compute())
        self.test_sample_metrics.reset()

        if self.any_train_samples:
            self.log_dict(self.train_sample_metrics.compute())
            self.train_sample_metrics.reset()
        else:
            log.info("No train samples generated at test time, so no metrics to log.")

    def configure_optimizers(self):
        opt = instantiate(self.optimizer, self.model.parameters())

        if self.lr_scheduler is None:
            return {"optimizer": opt}
        else:
            config = self.lr_scheduler.copy()
            config.pop("name")
            scheduler = instantiate(config, opt)
            lr_scheduler_config = {
                "interval": "step",
                "frequency": 1,
                "scheduler": scheduler,
            }
            return {"optimizer": opt, "lr_scheduler": lr_scheduler_config}

    def configure_callbacks(self):
        return [Plots()]

    def log_dict(self, *args, **kwargs):
        return super().log_dict(*args, **kwargs, add_dataloader_idx=False, sync_dist=True)
