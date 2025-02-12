import math

import torch
from jaxtyping import Float
from torch import Tensor, nn

from bsi.nn import MLP, FourierFeatures

from .pos_emb import NyquistPositionalEmbedding
from .utils import actfn_from_str


class DenoisingMLP(nn.Module):
    """A simple MLP denoising model."""

    def __init__(
        self,
        data_shape: tuple[int, ...],
        pos_emb: NyquistPositionalEmbedding,
        hidden_width: int,
        layers: int,
        actfn: str,
        zero_init: bool,
        fourier_features: FourierFeatures | None = None,
        **kwargs,
    ):
        super().__init__()

        self.data_shape = tuple(data_shape)
        self.pos_emb = pos_emb
        self.fourier_features = fourier_features

        n_dim = math.prod(data_shape)
        in_features = n_dim + self.pos_emb.size
        if fourier_features is not None:
            in_features += n_dim * fourier_features.n_features()
        self.layers = MLP(
            in_features,
            n_dim,
            hidden_features=hidden_width,
            hidden_layers=layers,
            actfn=actfn_from_str(actfn),
        )

        if zero_init:
            self.layers[-1].weight.data.zero_()
            self.layers[-1].bias.data.zero_()

    def forward(
        self, mu: Float[Tensor, "batch {self.data_shape}"], t: Float[Tensor, "batch"]
    ) -> Float[Tensor, "batch {self.data_shape}"]:
        x = [mu.flatten(start_dim=1), self.pos_emb(t)]
        if self.fourier_features is not None:
            x.append(self.fourier_features(x[0], dim=1))
        x = torch.cat(x, dim=-1)
        return self.layers(x).unflatten(dim=1, sizes=self.data_shape)
