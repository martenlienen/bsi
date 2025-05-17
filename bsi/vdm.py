import math

import einops as eo
import torch
import torch.distributions as td
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from .bsi import Discretization, broadcast_right


class VDM(nn.Module):
    """Implementation of Variational Diffusion Models. [1]

    [1] https://arxiv.org/abs/2107.00630
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        data_shape: tuple[int, ...],
        snr_min: float,
        snr_max: float,
        k: int,
        low_discrepancy_sampling: bool = True,
        discretization: Discretization | None = None,
    ):
        super().__init__()

        # Exclude model from state_dict here, so that we don't duplicate it in
        # checkpoints
        self._model = [model]
        self.data_shape = tuple(data_shape)
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.k = k
        self.low_discrepancy_sampling = low_discrepancy_sampling
        self.discretization = discretization

        self.register_buffer(
            "_gamma_0", -torch.as_tensor(snr_max).log(), persistent=False
        )
        self.register_buffer(
            "_gamma_1", -torch.as_tensor(snr_min).log(), persistent=False
        )

    @property
    def model(self):
        return self._model[0]

    def set_model(self, model):
        self._model[0] = model

    @property
    def tensor_args(self):
        return {"device": self._gamma_0.device, "dtype": self._gamma_0.dtype}

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
        l_prior = self.prior_loss(x)
        l_recon = self.reconstruction_loss(x, n_recon_samples, generator)
        l_diff = self.inf_diffusion_loss(x, n_measure_samples, generator)
        elbo = -(l_prior + l_recon.mean(dim=0) + l_diff.mean(dim=0))

        # Bits per dimension
        conversion_factor = -1 / (math.log(2) * math.prod(self.data_shape))
        bpd = conversion_factor * elbo

        extra = {"l_prior": l_prior, "l_recon": l_recon, "l_diff": l_diff}
        if estimate_var:
            # Estimate the variance of the Monte Carlo estimator
            assert n_recon_samples > 1 and n_measure_samples > 1, (
                "Need at least two samples of each to estimate variance"
            )
            l_recon_var = l_recon.var(dim=0, unbiased=True) / n_recon_samples
            l_diff_var = l_diff.var(dim=0, unbiased=True) / n_measure_samples
            extra["bpd_var"] = (conversion_factor**2) * (l_recon_var + l_diff_var)

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
        l_prior = self.prior_loss(x)
        l_recon = self.reconstruction_loss(x, n_recon_samples, generator)
        l_diff = self.finite_diffusion_loss(x, n_measure_samples, generator, t=t)
        elbo = -(l_prior + l_recon.mean(dim=0) + l_diff.mean(dim=0))

        # Bits per dimension
        conversion_factor = -1 / (math.log(2) * math.prod(self.data_shape))
        bpd = conversion_factor * elbo

        extra = {"l_prior": l_prior, "l_recon": l_recon, "l_diff": l_diff}
        if estimate_var:
            # Estimate the variance of the Monte Carlo estimator
            assert n_recon_samples > 1 and n_measure_samples > 1, (
                "Need at least two samples of each to estimate variance"
            )
            l_recon_var = l_recon.var(dim=0, unbiased=True) / n_recon_samples
            l_diff_var = l_diff.var(dim=0, unbiased=True) / n_measure_samples
            extra["bpd_var"] = (conversion_factor**2) * (l_recon_var + l_diff_var)

        return elbo, bpd, extra

    def prior_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
    ) -> Float[Tensor, "batch"]:
        var_1 = self.sigma2(x.new_ones((1,)))
        return 0.5 * eo.reduce(
            var_1 + (1 - var_1) * x.square() - torch.log(var_1) - 1,
            "batch ... -> batch",
            "sum",
        )

    def gamma(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.lerp(self._gamma_0, self._gamma_1, t)

    def sigma2(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.sigmoid(self.gamma(t))

    def alpha(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        # Compute alpha separately to avoid differences `1 - sigma2_t` close to 0 for
        # numerical stability
        return torch.sqrt(torch.sigmoid(-self.gamma(t)))

    def snr(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.exp(-self.gamma(t))

    def reconstruction_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "{n_samples} batch"]:
        """Sample the reconstruction loss of x."""
        zero = x.new_zeros((1,))
        alpha_0 = self.alpha(zero)
        mean = alpha_0 * x
        std = torch.sqrt(self.sigma2(zero))
        z_0 = torch.addcmul(
            mean,
            std,
            torch.randn(
                (n_samples, *x.shape), device=x.device, dtype=x.dtype, generator=generator
            ),
        )
        x_hat = z_0 / alpha_0

        p_recon = td.Normal(x_hat, std / alpha_0, validate_args=False)
        discretization = self.discretization
        if discretization is None:
            log_p_per_dim = p_recon.log_prob(x)
        else:
            # In contrast to BSI, VDM evaluates p_recon at the bin centers and then
            # normalizes to discretize the Normal distribution. This reduces the
            # reconstruction loss by about 0.03 bpd compared to the BSI discretization
            # for VDM. However, it requires memory linear in the number of
            # discretization bins which works fine for 8-bit images but would not work
            # for 16-bit audio, for example. Applying this discretization to BSI only
            # improves the reconstruction loss by about 0.002 bpd.
            boundaries = discretization.bin_boundaries(x.device, x.dtype)
            centers = (boundaries[1:] + boundaries[:-1]) / 2
            log_p_normal = p_recon.log_prob(broadcast_right(centers, x_hat[None]))
            log_p_discretized = F.log_softmax(log_p_normal, dim=0)
            x_idx = discretization.bucketize(x)
            log_p_per_dim = torch.gather(
                log_p_discretized,
                dim=0,
                index=eo.repeat(x_idx, "... -> 1 n ...", n=n_samples),
            )[0]

        return eo.reduce(-log_p_per_dim, "samples batch ... -> samples batch", "sum")

    def diffusion_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "batch"]:
        """Compute a Monte Carlo estimate of the diffusion loss in the finite step ELBO."""
        raise NotImplementedError()

    def finite_diffusion_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
        *,
        t: Float[Tensor, "{k + 1}"] | None = None,
    ) -> Float[Tensor, "{n_samples} batch"]:
        """Sample the diffusion loss in the finite step ELBO."""
        if t is None:
            t = torch.linspace(1.0, 0.0, self.k + 1, **self.tensor_args)

        T = len(t) - 1
        batch_size = len(x)
        i = torch.randint(
            0, T, (n_samples, batch_size), device=x.device, generator=generator
        )

        s_i = t[i + 1]
        t_i = t[i]

        z_t = self._sample_zt_given_x(x, t_i, generator)
        x_hat = self._predict_x(z_t.flatten(end_dim=1), t_i.flatten(end_dim=1))
        x_hat = eo.rearrange(x_hat, "(n b) ... -> n b ...", n=n_samples)
        decoding_error = eo.reduce((x - x_hat).square(), "n b ... -> n b", "sum")
        return 0.5 * T * (self.snr(s_i) - self.snr(t_i)) * decoding_error

    def inf_diffusion_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "{n_samples} batch"]:
        """Sample the diffusion loss in the infinite step ELBO."""
        t = self._sample_t(n_samples, len(x), generator)
        z_t = self._sample_zt_given_x(x, t, generator)
        x_hat = self._predict_x(z_t.flatten(end_dim=1), t.flatten(end_dim=1))
        x_hat = eo.rearrange(x_hat, "(n b) ... -> n b ...", n=n_samples)
        decoding_error = eo.reduce((x - x_hat).square(), "n b ... -> n b", "sum")

        # Since gamma is a linear function, we can compute the SNR gradient in closed
        # form
        dsnr_t_dt = -self.snr(t) * (self._gamma_0 - self._gamma_1)
        return 0.5 * dsnr_t_dt * decoding_error

    def train_loss(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        generator: torch.Generator | None = None,
    ):
        """Compute the training loss.

        The training loss is a single Monte Carlo sample of the infinite step ELBO with a
        mean over the data dimensions to make the magnitude of the loss independent of the
        data shape.
        """
        return self.inf_diffusion_loss(x, 1, generator) / math.prod(self.data_shape)

    def sample(
        self,
        n_samples: int,
        generator: torch.Generator | None = None,
        *,
        t: Float[Tensor, "{k + 1}"] | None = None,
    ) -> Float[Tensor, "{n_samples} {self.data_shape}"]:
        """Draw n_samples samples."""
        if t is None:
            ts = torch.linspace(1.0, 0.0, self.k + 1, **self.tensor_args)
        else:
            ts = t

        z_t = torch.randn(
            (n_samples, *self.data_shape), **self.tensor_args, generator=generator
        )

        for t, s in zip(ts[:-1], ts[1:]):
            # There is a compiler bug in pytorch 2.6 where t and s take on unexpected
            # values if we don't clone them before repeating.
            t = eo.repeat(t.clone(), "-> n", n=n_samples)
            s = eo.repeat(s.clone(), "-> n", n=n_samples)
            x_hat = self._predict_x(z_t, t)
            z_t = self._sample_zs_given_zt_x(s, z_t, t, x_hat, generator)

        alpha_0 = self.alpha(z_t.new_zeros((1,)))
        return z_t / alpha_0

    def sample_history(
        self,
        n_samples: int,
        generator: torch.Generator | None = None,
        *,
        t: Float[Tensor, "{k + 1}"] | None = None,
    ) -> Float[Tensor, "{self.k + 1} {n_samples} {self.data_shape}"]:
        """Draw n_samples samples."""
        if t is None:
            ts = torch.linspace(1.0, 0.0, self.k + 1, **self.tensor_args)
        else:
            ts = t

        x_hats = torch.zeros(
            (self.k + 1, n_samples, *self.data_shape), **self.tensor_args
        )
        z_t = torch.randn(
            (n_samples, *self.data_shape), **self.tensor_args, generator=generator
        )

        for i, (t, s) in enumerate(zip(ts[:-1], ts[1:])):
            # There is a compiler bug in pytorch 2.6 where t and s take on unexpected
            # values if we don't clone them before repeating.
            t = eo.repeat(t.clone(), "-> n", n=n_samples)
            s = eo.repeat(s.clone(), "-> n", n=n_samples)
            x_hats[i] = self._predict_x(z_t, t)
            z_t = self._sample_zs_given_zt_x(s, z_t, t, x_hats[i], generator)

        alpha_0 = self.alpha(z_t.new_zeros((1,)))
        x_hats[-1] = z_t / alpha_0
        return x_hats

    def _predict_x(
        self, z_t: Float[Tensor, "batch {self.data_shape}"], t: Float[Tensor, "batch"]
    ) -> Float[Tensor, "batch {self.data_shape}"]:
        return (
            z_t - broadcast_right(torch.sqrt(self.sigma2(t)), z_t) * self.model(z_t, t)
        ) / broadcast_right(self.alpha(t), z_t)

    def _sample_zt_given_x(
        self,
        x: Float[Tensor, "batch {self.data_shape}"],
        t: Float[Tensor, "... batch"],
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "... batch {self.data_shape}"]:
        x = x[*((None,) * (t.ndim - 1))]
        eps = torch.randn(
            (*t.shape, *self.data_shape),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        return torch.addcmul(
            broadcast_right(self.alpha(t), x) * x,
            broadcast_right(torch.sqrt(self.sigma2(t)), x),
            eps,
        )

    def _sample_zs_given_zt_x(
        self,
        s: Float[Tensor, "batch"],
        z_t: Float[Tensor, "batch {self.data_shape}"],
        t: Float[Tensor, "batch"],
        x: Float[Tensor, "batch {self.data_shape}"],
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "batch {self.data_shape}"]:
        # Compute most things in log-space to avoid numerically unstable differences and
        # divisions as the authors describe in the paper
        g_s = self.gamma(s)
        g_t = self.gamma(t)
        sigma2_ts_over_sigma2_t = -torch.expm1(
            F.softplus(-g_t) - F.softplus(g_t) - F.softplus(-g_s) + F.softplus(g_s)
        )
        mean = (
            broadcast_right(
                torch.exp(
                    0.5 * (F.softplus(g_s) - F.softplus(g_t))
                    + F.softplus(-g_t)
                    - F.softplus(-g_s)
                ),
                z_t,
            )
            * z_t
            + broadcast_right(self.alpha(s) * sigma2_ts_over_sigma2_t, x) * x
        )
        std = torch.sqrt(self.sigma2(s) * sigma2_ts_over_sigma2_t)
        eps = torch.randn(z_t.shape, dtype=x.dtype, device=x.device, generator=generator)
        return torch.addcmul(mean, broadcast_right(std, eps), eps)

    def _sample_t(
        self, n_samples: int, batch_size: int, generator: torch.Generator | None = None
    ) -> Float[Tensor, "{n_samples} {batch_size}"]:
        if self.low_discrepancy_sampling:
            # Low-discrepancy sampling as in the VDM paper [1].
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
