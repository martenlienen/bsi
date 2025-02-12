from functools import partial

import torch
from jaxtyping import Float
from torch import Tensor, nn

from bsi.nn import (
    Attention2D,
    FourierFeatures,
    KwargsSequential,
    Residual,
    ResidualBlock,
    SimplifiedUNet,
)

from .pos_emb import NyquistPositionalEmbedding
from .utils import actfn_from_str


class DenoisingVDMUNet(nn.Module):
    """U-Net structure as in the VDM paper without downsampling."""

    def __init__(
        self,
        data_shape: tuple[int, ...],
        pos_emb: NyquistPositionalEmbedding,
        actfn: str,
        dim: int,
        levels: int,
        pos_emb_mult: int,
        n_attention_heads: int = 1,
        dropout: float | None = None,
        downsampling_attention: bool = False,
        fourier_features: FourierFeatures | None = None,
        padding_mode: str = "zeros",
        **kwargs,
    ):
        super().__init__()

        self.data_shape = tuple(data_shape)
        self.pos_emb = pos_emb
        self.fourier_features = fourier_features

        assert len(self.data_shape) == 3, "Only works for 2D images"

        n_channels = data_shape[0]
        in_features = out_features = n_channels
        if fourier_features is not None:
            in_features += n_channels * fourier_features.n_features()

        ActFn = actfn_from_str(actfn)
        Norm = partial(nn.GroupNorm, 32)
        residual_block = partial(
            ResidualBlock,
            ActFn=ActFn,
            Norm=Norm,
            dropout=dropout,
            attention=downsampling_attention,
            padding_mode=padding_mode,
        )

        c_dim = pos_emb.size * pos_emb_mult
        self.pos_map = KwargsSequential(
            self.pos_emb,
            nn.Linear(pos_emb.size, c_dim),
            ActFn(),
            nn.Linear(c_dim, c_dim),
            ActFn(),
        )

        self.encode = nn.Conv2d(in_features, dim, 3, padding=1, padding_mode=padding_mode)
        self.decode = nn.Conv2d(dim, out_features, 1)

        downsampling_blocks = [
            residual_block(dim, dim, c_dim=c_dim) for i in range(levels)
        ]
        upsampling_blocks = [
            residual_block(2 * dim, dim, c_dim=c_dim) for i in range(levels)
        ]
        center_block = KwargsSequential(
            residual_block(dim, dim, c_dim=c_dim),
            Residual(
                KwargsSequential(
                    Norm(dim),
                    Attention2D(dim, heads=n_attention_heads, padding_mode=padding_mode),
                )
            ),
            residual_block(dim, dim, c_dim=c_dim),
        )
        self.u_net = SimplifiedUNet(downsampling_blocks, upsampling_blocks, center_block)

    def forward(
        self, mu: Float[Tensor, "batch {self.data_shape}"], t: Float[Tensor, "batch"]
    ) -> Float[Tensor, "batch {self.data_shape}"]:
        x_parts = [mu]
        if self.fourier_features is not None:
            x_parts.append(self.fourier_features(mu, dim=1))
        x = torch.cat(x_parts, dim=1)

        return self.decode(self.u_net(self.encode(x), c=self.pos_map(t)))
