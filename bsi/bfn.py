import math

import einops as eo
import torch
import torch.distributions as td
from jaxtyping import Float
from torch import Tensor, nn

from .bsi import Discretization, broadcast_right


class BFN(nn.Module):
    """Implementation of Bayesian Flow Networks. [1]

    [1] https://arxiv.org/abs/2308.07037
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        data_shape: tuple[int, ...],
        sigma_1: float,
        k: int,
        x_min: float = -1.0,
        x_max: float = 1.0,
        t_min: float = 1e-6,
        low_discrepancy_sampling: bool = True,
        discretization: Discretization | None = None,
    ):
        super().__init__()

        # Exclude model from state_dict here, so that we don't duplicate it in
        # checkpoints
        self._model = [model]
        self.data_shape = tuple(data_shape)

        assert sigma_1 < 1.0, "`sigma_1 < 1` is required by BFN formulas"
        self.register_buffer("sigma_1", torch.as_tensor(sigma_1), persistent=False)

        self.k = k
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.low_discrepancy_sampling = low_discrepancy_sampling
        self.discretization = discretization

    @property
    def model(self):
        return self._model[0]

    def set_model(self, model):
        self._model[0] = model

    @property
    def tensor_args(self):
        return {"device": self.sigma_1.device, "dtype": self.sigma_1.dtype}

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
        l_latent = self.continuous_time_loss(x, n_measure_samples, generator)
        elbo = -(l_recon.mean(dim=0) + l_latent.mean(dim=0))

        # Bits per dimension
        conversion_factor = -1 / (math.log(2) * math.prod(self.data_shape))
        bpd = conversion_factor * elbo

        extra = {"l_recon": l_recon, "l_latent": l_latent}
        if estimate_var:
            # Estimate the variance of the Monte Carlo estimator
            assert n_recon_samples > 1 and n_measure_samples > 1, (
                "Need at least two samples of each to estimate variance"
            )
            l_recon_var = l_recon.var(dim=0, unbiased=True) / n_recon_samples
            l_latent_var = l_latent.var(dim=0, unbiased=True) / n_measure_samples
            extra["bpd_var"] = (conversion_factor**2) * (l_recon_var + l_latent_var)

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
        l_latent = self.discrete_time_loss(x, n_measure_samples, generator, t=t)
        elbo = -(l_recon.mean(dim=0) + l_latent.mean(dim=0))

        # Bits per dimension
        conversion_factor = -1 / (math.log(2) * math.prod(self.data_shape))
        bpd = conversion_factor * elbo

        extra = {"l_recon": l_recon, "l_latent": l_latent}
        if estimate_var:
            # Estimate the variance of the Monte Carlo estimator
            assert n_recon_samples > 1 and n_measure_samples > 1, (
                "Need at least two samples of each to estimate variance"
            )
            l_recon_var = l_recon.var(dim=0, unbiased=True) / n_recon_samples
            l_latent_var = l_latent.var(dim=0, unbiased=True) / n_measure_samples
            extra["bpd_var"] = (conversion_factor**2) * (l_recon_var + l_latent_var)

        return elbo, bpd, extra

    def reconstruction_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "batch"]:
        """Sample the reconstruction loss of x."""
        t = x.new_ones((n_samples, len(x)))
        mu = self._sample_flow_distribution(x, t, generator)
        x_hat = self._predict_x(mu.flatten(end_dim=1), t.flatten(end_dim=1))
        x_hat = eo.rearrange(x_hat, "(n b) ... -> n b ...", n=n_samples)

        p_recon = td.Normal(
            x_hat, torch.full_like(x_hat, self.sigma_1), validate_args=False
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

    def discrete_time_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
        *,
        t: Float[Tensor, "{k + 1}"] | None = None,
    ) -> Float[Tensor, "{n_samples} batch"]:
        """Sample the discrete time loss."""
        if t is None:
            t = self.linspace(0, 1, self.k + 1, **self.tensor_args)

        n = len(t) - 1
        batch_size = len(x)
        i = torch.randint(
            0, n, (n_samples, batch_size), device=x.device, generator=generator
        )

        t_i = t[i]
        mu = self._sample_flow_distribution(x, t_i, generator)
        x_hat = self._predict_x(mu.flatten(end_dim=1), t_i.flatten(end_dim=1))
        x_hat = eo.rearrange(x_hat, "(n b) ... -> n b ...", n=n_samples)
        decoding_error = eo.reduce(
            (x - x_hat).square(), "sample batch ... -> sample batch", "sum"
        )
        return (
            0.5
            * n
            * (1 - (self.sigma_1 ** (2 / n)))
            * ((self.sigma_1 ** ((-2 / n) * (i + 1))) * decoding_error)
        )

    def continuous_time_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "{n_samples} batch"]:
        """Sample the continuous time loss."""
        t = self._sample_t(n_samples, len(x), generator)
        mu = self._sample_flow_distribution(x, t, generator)
        x_hat = self._predict_x(mu.flatten(end_dim=1), t.flatten(end_dim=1))
        x_hat = eo.rearrange(x_hat, "(n b) ... -> n b ...", n=n_samples)
        decoding_error = eo.reduce(
            (x - x_hat).square(), "sample batch ... -> sample batch", "sum"
        )
        return -torch.log(self.sigma_1) * ((self.sigma_1 ** (-2 * t)) * decoding_error)

    def train_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        generator: torch.Generator | None = None,
    ):
        """Compute the training loss.

        This is the same as the continuous time loss but with just a single sample, no
        constant factors and a mean over data dimensions instead of sum, to make the
        magnitude of the loss independent of the number of data dimensions.
        """

        t = self._sample_t(1, len(x), generator)[0]
        mu = self._sample_flow_distribution(x, t, generator)
        x_hat = self._predict_x(mu, t)
        decoding_error = eo.reduce((x - x_hat).square(), "batch ... -> batch", "mean")
        return ((self.sigma_1 ** (-2 * t)) * decoding_error).mean(dim=0)

    def sample(
        self,
        n_samples: int,
        generator: torch.Generator | None = None,
        *,
        t: Float[Tensor, "{k + 1}"] | None = None,
    ) -> Float[Tensor, "{n_samples} {self.data_shape}"]:
        """Draw n_samples samples."""
        if t is None:
            t = torch.linspace(0, 1, self.k + 1, **self.tensor_args)
        n = len(t) - 1
        mu = torch.zeros((n_samples, *self.data_shape), **self.tensor_args)
        rho = 1.0
        for i in range(n):
            x_hat = self._predict_x(mu, eo.repeat(t[i].clone(), "-> n", n=n_samples))
            alpha = self.sigma_1 ** (-2 * t[i + 1]) * (
                1 - self.sigma_1 ** (2 * (t[i + 1] - t[i]))
            )
            y = x_hat + torch.rsqrt(alpha) * torch.randn(
                (n_samples, *self.data_shape), **self.tensor_args, generator=generator
            )
            mu = (rho * mu + alpha * y) / (rho + alpha)
            rho = rho + alpha
        return self._predict_x(mu, mu.new_ones((n_samples,)))

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
        """Draw n_samples samples."""
        if t is None:
            t = torch.linspace(0, 1, self.k + 1, **self.tensor_args)
        n = len(t) - 1
        x_hats = torch.zeros((n + 1, n_samples, *self.data_shape), **self.tensor_args)
        mu = torch.zeros((n_samples, *self.data_shape), **self.tensor_args)
        rho = 1.0
        mus = [mu]
        ys = []
        for i in range(n):
            x_hats[i] = self._predict_x(mu, eo.repeat(t[i].clone(), "-> n", n=n_samples))
            alpha = self.sigma_1 ** (-2 * t[i + 1]) * (
                1 - self.sigma_1 ** (2 * (t[i + 1] - t[i]))
            )
            y = x_hats[i] + torch.rsqrt(alpha) * torch.randn(
                (n_samples, *self.data_shape), **self.tensor_args, generator=generator
            )
            mu = (rho * mu + alpha * y) / (rho + alpha)
            rho = rho + alpha

            ys.append(y)
            mus.append(mu)
        x_hats[-1] = self._predict_x(mu, mu.new_ones((n_samples,)))
        return torch.stack(mus), x_hats, torch.stack(ys)

    def _predict_x(
        self, mu: Float[Tensor, "batch {self.data_shape}"], t: Float[Tensor, "batch"]
    ) -> Float[Tensor, "batch {self.data_shape}"]:
        eps_hat = self.model(mu, t)
        gamma = 1 - self.sigma_1 ** (2 * torch.clamp(t, min=self.t_min))
        x_hat = (
            mu / broadcast_right(gamma, mu)
            - broadcast_right(torch.sqrt((1 - gamma) / gamma), eps_hat) * eps_hat
        ).clip(self.x_min, self.x_max)
        return torch.where(broadcast_right(t < self.t_min, x_hat), 0, x_hat)

    def _sample_flow_distribution(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        t: Float[Tensor, "... batch"],
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "... batch {self.data_shape}"]:
        x = x[*((None,) * (t.ndim - 1))]
        gamma = 1 - self.sigma_1 ** (2 * t)
        return torch.addcmul(
            broadcast_right(gamma, x) * x,
            broadcast_right(torch.sqrt(gamma * (1 - gamma)), x),
            torch.randn(
                (*t.shape, *self.data_shape), **self.tensor_args, generator=generator
            ),
        )

    def _sample_t(
        self, n_samples: int, batch_size: int, generator: torch.Generator | None = None
    ) -> Float[Tensor, "{n_samples} {batch_size}"]:
        if self.low_discrepancy_sampling:
            # Low-discrepancy sampling of the interval [0, 1] as in the VDM paper [1].
            #
            # [1] https://arxiv.org/abs/2107.00630
            offset = torch.rand((), **self.tensor_args, generator=generator)
            total = n_samples * batch_size
            # Random permutation of `i / (1 + total)` so that a batch element is not
            # evaluated at consecutive lambdas
            grid = torch.randperm(
                total, device=self.tensor_args["device"], generator=generator
            ) / (1 + total)
            return torch.remainder(
                eo.rearrange(grid, "(n b) -> n b", n=n_samples) + offset, 1
            )
        else:
            return torch.rand(
                (batch_size, n_samples), **self.tensor_args, generator=generator
            )
