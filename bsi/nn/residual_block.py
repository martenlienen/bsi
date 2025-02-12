import torch
from torch import nn

from .attention import Attention2D
from .sequential import KwargsSequential


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()

        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)


class FeatureModulation(nn.Module):
    def forward(self, x, *, scale_shift):
        scale, shift = scale_shift
        return torch.addcmul(shift[..., None, None], scale[..., None, None] + 1, x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        c_dim: int,
        ActFn,
        Norm,
        dropout,
        attention: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.project_onto_scale_shift = nn.Linear(c_dim, dim_out * 2, 1)
        self.skip = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.layers = KwargsSequential(
            Norm(dim_in),
            ActFn(),
            nn.Conv2d(dim_in, dim_out, 3, padding=1, padding_mode=padding_mode),
            FeatureModulation(),
            ActFn(),
            *([nn.Dropout(dropout)] if dropout is not None else []),
            nn.Conv2d(dim_out, dim_out, 3, padding=1, padding_mode=padding_mode),
        )

        self.attention = attention
        if attention:
            self.res_attention = Residual(
                KwargsSequential(
                    Norm(dim_out), Attention2D(dim_out, padding_mode=padding_mode)
                )
            )
        else:
            self.res_attention = nn.Identity()

    def forward(self, x, c):
        scale_shift = self.project_onto_scale_shift(c).chunk(2, dim=1)
        x = self.skip(x) + self.layers(x, scale_shift=scale_shift)
        return self.res_attention(x)
