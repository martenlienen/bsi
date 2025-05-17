import math
from dataclasses import dataclass
from typing import Literal

import einops as eo
import torch
import torch.distributions as td
from jaxtyping import Float, Int, UInt8
from torch import Tensor, nn


@dataclass
class Discretization:
    """A discretization of the interval [min, max] into k bins.

    The bins are open on the right and centered on `min + (max - min) * (i - 1) / (k -
    1)` for `i = 1..k`.
    """

    min: float
    max: float
    k: int

    @classmethod
    def image_8bit(cls):
        """Discretization of 8-bit images rescaled to the [-1, 1] interval."""
        return cls(-1.0, 1.0, 256)

    def bin_boundaries(self, device: torch.device, dtype: torch.dtype):
        return torch.linspace(*self.range, self.k + 1, device=device, dtype=dtype)

    def bucketize(self, x: Float[Tensor, "..."]) -> Int[Tensor, "..."]:
        """Find the discrete bucket index of continuous values in the [min, max] range."""
        dx = self.dx
        return ((x - (self.min - dx / 2)) / dx).to(torch.int64).clamp(0, self.k - 1)

    def to_unit_interval(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """Map x from [min, max] to [0, 1]."""
        return (x - self.min) / (self.max - self.min)

    def to_8bit_image(self, data) -> UInt8[Tensor, "..."]:
        """Convert continuous data in the [min, max] range into a discretized 8-bit image."""
        uint8 = torch.iinfo(torch.uint8)
        return (
            (self.to_unit_interval(data) * 255)
            .clamp(uint8.min, uint8.max)
            .to(torch.uint8)
        )

    @property
    def range(self) -> tuple[float, float]:
        dx = self.dx
        return (self.min - dx / 2, self.max + dx / 2)

    @property
    def dx(self) -> float:
        """Width of a single bin."""
        return (self.max - self.min) / (self.k - 1)


def broadcast_right(x: torch.Tensor, other: torch.Tensor):
    """Unsqueeze `x` to the right so that it broadcasts against `other`."""
    assert other.ndim >= x.ndim
    return x.reshape(*x.shape, *((1,) * (other.ndim - x.ndim)))


class LogUniform:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

        self.ln_low = math.log(self.low)
        self.ln_high = math.log(self.high)
        self.diff_ln_high_ln_low = self.ln_high - self.ln_low

    def reciprocal_pdf(self, value: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """Return the reciprocal probability density at `value`."""
        return value * self.diff_ln_high_ln_low

    def cdf(self, value: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return (torch.log(value) - self.ln_low) / self.diff_ln_high_ln_low

    def icdf(self, quantile: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.exp(self.diff_ln_high_ln_low * quantile + self.ln_low)


class BSI(nn.Module):
    """Implementation of Bayesian Sample Inference [1].

    [1] https://arxiv.org/abs/2502.07580
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        data_shape: tuple[int, ...],
        lambda_0: float,
        alpha_M: float,
        alpha_R: float,
        k: int,
        preconditioning: Literal["edm"] | None,
        low_discrepancy_sampling: bool = True,
        discretization: Discretization | None = None,
    ):
        r"""
        Args:
            model: Model `f_\theta` taking two arguments, a batch of noisy data samples
              `\mu` and a batch of scalar noise levels `t \in [0, 1]`.
            data_shape: Data shape, e.g. [3, 32, 32] for ImageNet32
            lambda_0: Initial belief precision, e.g. 1e-2 as a good default for normalized data
            alpha_M: Maximum measurement precision, e.g. 1e6 as a good default for normalized data
            alpha_R: Reconstruction precision. 2*alpha_M gives good likelihoods
            k: Number of default sampling steps.
            preconditioning: Whether to use the EDM-like preconditioning derived in the paper.
            low_discrepancy_sampling: Whether to use low-discrepancy sampling to smooth out the train loss.
            discretization: Discretization for computing likelihoods in bits per dimension.
              Use`Discretization.image_8bit()` for 8-bit images.
        """

        super().__init__()

        # Exclude model from state_dict here, so that we don't duplicate it in
        # checkpoints
        self._model = [model]
        self.data_shape = tuple(data_shape)
        self.register_buffer("lambda_0", torch.as_tensor(lambda_0), persistent=False)
        self.register_buffer("alpha_R", torch.as_tensor(alpha_R), persistent=False)
        self.register_buffer("alpha_M", torch.as_tensor(alpha_M), persistent=False)
        self.k = k
        self.preconditioning = preconditioning
        self.low_discrepancy_sampling = low_discrepancy_sampling
        self.discretization = discretization

        self.p_lambda = LogUniform(self.lambda_0, self.lambda_0 + self.alpha_M)

        self.register_buffer(
            "default_schedule", torch.linspace(0.0, 1.0, self.k + 1), persistent=False
        )

    @property
    def model(self):
        return self._model[0]

    def set_model(self, model):
        self._model[0] = model

    @property
    def tensor_args(self):
        return {"device": self.lambda_0.device, "dtype": self.lambda_0.dtype}

    def elbo(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_recon_samples: int,
        n_measure_samples: int,
        generator: torch.Generator | None = None,
        *,
        estimate_var: bool = False,
    ) -> tuple[
        Float[Tensor, "batch"], Float[Tensor, "batch"], dict[str, Float[Tensor, "batch"]]
    ]:
        """Compute a Monte Carlo estimate of the infinite step ELBO."""
        l_recon = self.reconstruction_loss(x, n_recon_samples, generator)
        l_measure = self.inf_measurement_loss(x, n_measure_samples, generator)
        elbo = -(l_recon.mean(dim=0) + l_measure.mean(dim=0))

        # Bits per dimension
        conversion_factor = -1 / (math.log(2) * math.prod(self.data_shape))
        bpd = conversion_factor * elbo

        extra = {"l_recon": l_recon, "l_measure": l_measure}
        if estimate_var:
            # Estimate the variance of the Monte Carlo estimator
            assert n_recon_samples > 1 and n_measure_samples > 1, (
                "Need at least two samples of each to estimate variance"
            )
            l_recon_var = l_recon.var(dim=0, unbiased=True) / n_recon_samples
            l_measure_var = l_measure.var(dim=0, unbiased=True) / n_measure_samples
            extra["bpd_var"] = (conversion_factor**2) * (l_recon_var + l_measure_var)

        return elbo, bpd, extra

    def finite_elbo(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_recon_samples: int,
        n_measure_samples: int,
        generator: torch.Generator | None = None,
        *,
        t: Float[Tensor, "{k + 1}"] | None = None,
        estimate_var: bool = False,
    ) -> tuple[
        Float[Tensor, "batch"], Float[Tensor, "batch"], dict[str, Float[Tensor, "batch"]]
    ]:
        """Compute a Monte Carlo estimate of the finite step ELBO."""
        l_recon = self.reconstruction_loss(x, n_recon_samples, generator)
        l_measure = self.finite_measurement_loss(x, n_measure_samples, generator, t=t)
        elbo = -(l_recon.mean(dim=0) + l_measure.mean(dim=0))

        # Bits per dimension
        conversion_factor = -1 / (math.log(2) * math.prod(self.data_shape))
        bpd = conversion_factor * elbo

        extra = {"l_recon": l_recon, "l_measure": l_measure}
        if estimate_var:
            # Estimate the variance of the Monte Carlo estimator
            assert n_recon_samples > 1 and n_measure_samples > 1, (
                "Need at least two samples of each to estimate variance"
            )
            l_recon_var = l_recon.var(dim=0, unbiased=True) / n_recon_samples
            l_measure_var = l_measure.var(dim=0, unbiased=True) / n_measure_samples
            extra["bpd_var"] = (conversion_factor**2) * (l_recon_var + l_measure_var)

        return elbo, bpd, extra

    def reconstruction_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "{n_samples} batch"]:
        """Sample the reconstruction loss of x."""
        mu_lambda_M = self._sample_q_mu_lambda(
            x, x.new_full((n_samples, len(x)), self.lambda_0 + self.alpha_M), generator
        ).flatten(end_dim=1)
        x_hat = self._predict_x(mu_lambda_M, x.new_ones(n_samples * len(x)))
        x_hat = eo.rearrange(x_hat, "(n b) ... -> n b ...", n=n_samples)

        p_recon = td.Normal(
            x_hat, torch.full_like(x_hat, torch.rsqrt(self.alpha_R)), validate_args=False
        )
        discretization = self.discretization
        if discretization is None:
            log_p_per_dim = p_recon.log_prob(x)
        else:
            boundaries = discretization.bin_boundaries(x.device, x.dtype)
            x_idx = discretization.bucketize(x)
            cdf_left = p_recon.cdf(boundaries[x_idx])
            cdf_right = p_recon.cdf(boundaries[x_idx + 1])
            cdf_left_clamped = torch.where(x_idx == 0, 0, cdf_left)
            cdf_right_clamped = torch.where(x_idx == discretization.k - 1, 1, cdf_right)
            log_p_per_dim = torch.log(
                torch.clamp(cdf_right_clamped - cdf_left_clamped, min=1e-20)
            )

        return eo.reduce(-log_p_per_dim, "samples batch ... -> samples batch", "sum")

    def finite_measurement_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
        *,
        t: Float[Tensor, "{k + 1}"] | None = None,
    ) -> Float[Tensor, "{n_samples} batch"]:
        """Sample the measurement loss in the finite step ELBO."""
        if t is None:
            t = self.default_schedule

        lambda_ = self.p_lambda.icdf(t)
        alpha = lambda_.diff()

        batch_size = len(x)
        k = len(alpha)
        i = torch.randint(
            0, k, (n_samples, batch_size), device=x.device, generator=generator
        )

        mu_lambda = self._sample_q_mu_lambda(x, lambda_[i], generator)
        x_hat = self._predict_x(mu_lambda.flatten(end_dim=1), t[i].flatten(end_dim=1))
        x_hat = eo.rearrange(x_hat, "(n b) ... -> n b ...", n=n_samples)
        decoding_error = eo.reduce((x - x_hat).square(), "n b ... -> n b", "sum")
        return (0.5 * k) * alpha[i] * decoding_error

    def inf_measurement_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "batch"]:
        """Sample the measurement loss in the infinite step ELBO."""
        lambda_ = self._sample_lambda(n_samples, len(x), generator)
        mu_lambda = self._sample_q_mu_lambda(x, lambda_, generator)
        t = self.p_lambda.cdf(lambda_).flatten(end_dim=1)
        x_hat = self._predict_x(mu_lambda.flatten(end_dim=1), t)
        x_hat = eo.rearrange(x_hat, "(n b) ... -> n b ...", n=n_samples)
        decoding_error = eo.reduce((x - x_hat).square(), "n b ... -> n b", "sum")
        return 0.5 * self.p_lambda.reciprocal_pdf(lambda_) * decoding_error

    def train_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "batch"]:
        """Compute the training loss.

        The training loss is just the infinite-step ELBO with

            - mean instead of sum over the dimensions to make the magnitude of the loss
              independent of the data shape
            - no constant scaling factors
            - a single Monte Carlo sample per data sample
        """

        lambda_ = self._sample_lambda(1, len(x), generator)[0]
        mu = self._sample_q_mu_lambda(x, lambda_, generator)
        x_hat = self._predict_x(mu, self.p_lambda.cdf(lambda_))
        decoding_error = eo.reduce((x - x_hat).square(), "batch ... -> batch", "mean")
        return self.p_lambda.reciprocal_pdf(lambda_) * decoding_error

    def sample(
        self,
        n_samples: int,
        generator: torch.Generator | None = None,
        *,
        t: Float[Tensor, "{k + 1}"] | None = None,
    ) -> Float[Tensor, "{n_samples} {self.data_shape}"]:
        """Draw `n_samples` samples."""
        if t is None:
            t = self.default_schedule
        lambda_ = self.p_lambda.icdf(t)
        alpha = lambda_.diff()
        k = len(alpha)
        mu = torch.rsqrt(lambda_[0]) * torch.randn(
            (n_samples, *self.data_shape), **self.tensor_args, generator=generator
        )
        for i in range(k):
            # There is a compiler bug in pytorch 2.6 where t[i] takes on unexpected
            # values if we don't clone it before repeating.
            x_hat = self._predict_x(mu, eo.repeat(t[i].clone(), "-> n", n=n_samples))
            y = x_hat + torch.rsqrt(alpha[i]) * torch.randn(
                (n_samples, *self.data_shape), **self.tensor_args, generator=generator
            )
            mu = (alpha[i] * y + lambda_[i] * mu) / lambda_[i + 1]
        return self._predict_x(mu, mu.new_ones(n_samples))

    def sample_history(
        self,
        n_samples: int,
        generator: torch.Generator | None = None,
        *,
        t: Float[Tensor, "{k + 1}"] | None = None,
    ) -> tuple[
        Float[Tensor, "{k + 1} {n_samples} {self.data_shape}"],
        Float[Tensor, "{k + 1} {n_samples} {self.data_shape}"],
        Float[Tensor, "{k} {n_samples} {self.data_shape}"],
    ]:
        """Draw `n_samples` samples and return all intermediate steps."""
        if t is None:
            t = self.default_schedule
        lambda_ = self.p_lambda.icdf(t)
        alpha = lambda_.diff()
        k = len(alpha)
        x_hats = torch.zeros((k + 1, n_samples, *self.data_shape), **self.tensor_args)
        mu = torch.rsqrt(lambda_[0]) * torch.randn(
            (n_samples, *self.data_shape), **self.tensor_args, generator=generator
        )
        mus = [mu]
        ys = []
        for i in range(k):
            # There is a compiler bug in pytorch 2.6 where t[i] takes on unexpected
            # values if we don't clone it before repeating.
            x_hats[i] = self._predict_x(mu, eo.repeat(t[i].clone(), "-> n", n=n_samples))
            y = x_hats[i] + torch.rsqrt(alpha[i]) * torch.randn(
                (n_samples, *self.data_shape), **self.tensor_args, generator=generator
            )
            mu = (alpha[i] * y + lambda_[i] * mu) / lambda_[i + 1]

            ys.append(y)
            mus.append(mu)
        x_hats[-1] = self._predict_x(mu, mu.new_ones(n_samples))
        return torch.stack(mus), x_hats, torch.stack(ys)

    def _predict_x(
        self, mu: Float[Tensor, "batch {self.data_shape}"], t: Float[Tensor, "batch"]
    ) -> Float[Tensor, "batch {self.data_shape}"]:
        if self.preconditioning is None:
            return self.model(mu, t)
        elif self.preconditioning == "edm":
            c_skip, c_out, c_in = self._edm_preconditioning(t)
            return torch.addcmul(
                broadcast_right(c_skip, mu) * mu,
                broadcast_right(c_out, mu),
                self.model(broadcast_right(c_in, mu) * mu, t),
            )
        else:
            raise RuntimeError(f"Unknown preconditioning {self.preconditioning}")

    def _edm_preconditioning(self, t: Float[Tensor, "batch"] | None = None):
        """Preconditioning derived as in the EDM paper [1].

        [1] https://arxiv.org/abs/2206.00364
        """

        lambda_ = self.p_lambda.icdf(t)
        alpha = lambda_ - self.lambda_0
        # Written in this way to avoid squaring alpha for float stability
        kappa = 1 + alpha * (alpha / lambda_)
        c_skip = alpha / kappa
        c_out = torch.rsqrt(kappa)
        c_in = torch.sqrt(lambda_ / kappa)
        return c_skip, c_out, c_in

    def _sample_q_mu_lambda(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        lambda_: Float[Tensor, "... batch"],
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "... batch {self.data_shape}"]:
        x = x[*((None,) * (lambda_.ndim - 1))]
        return torch.addcmul(
            broadcast_right((lambda_ - self.lambda_0) / lambda_, x) * x,
            broadcast_right(torch.rsqrt(lambda_), x),
            torch.randn(
                (*lambda_.shape, *self.data_shape),
                **self.tensor_args,
                generator=generator,
            ),
        )

    def _sample_lambda(
        self, n_samples: int, batch_size: int, generator: torch.Generator | None = None
    ) -> Float[Tensor, "{n_samples} {batch_size}"]:
        if self.low_discrepancy_sampling:
            # Low-discrepancy sampling as in the VDM paper [1]. We sample the interval
            # [0, 1] and then transform to alpha samples with the inverse-CDF method.
            #
            # [1] https://arxiv.org/abs/2107.00630
            offset = torch.rand((), **self.tensor_args, generator=generator)
            total = n_samples * batch_size
            # Random permutation of `i / (1 + total)` so that a batch element is not
            # evaluated at consecutive lambdas
            grid = torch.randperm(
                total, device=self.tensor_args["device"], generator=generator
            ) / (1 + total)
            t = torch.remainder(
                eo.rearrange(grid, "(n b) -> n b", n=n_samples) + offset, 1
            )
            return self.p_lambda.icdf(t)
        else:
            t = torch.rand(
                (batch_size, n_samples), **self.tensor_args, generator=generator
            )
            return self.p_lambda.icdf(t)
