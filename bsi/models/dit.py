"""
Code adapted from the official implementation [1].

Should almost be functionally the same, but has shape annotations, no unused code
brances and does not rely on the timm library. In addition, we have added a linear layer
before the SiLU when computing the addLN modulation and dropout in the DiT block before
the MLP to prevent overfitting on small datasets.

[1] https://github.com/facebookresearch/DiT
"""

import math
from functools import partial

import einops as eo
import torch
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor, nn

from bsi.nn import MLP, FourierFeatures

from .pos_emb import NyquistPositionalEmbedding


def scaled_dot_product_attention(
    query: Float[Tensor, "batch heads patch channels"],
    key: Float[Tensor, "batch heads patch channels"],
    value: Float[Tensor, "batch heads patch channels"],
    dropout_p: float = 0.0,
) -> Tensor:
    """Non-fused attention.

    We have had problems with the fused attention kernels sometimes producing NaN
    gradients even in non-degenerate conditions, so now we try this.

    This is the pure-pytorch implementation equivalent to the efficient-attention fused
    kernel according to the docstring of `F.scaled_dot_product_attention`.
    """

    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim: int, *, heads: int, dropout: float = 0.0):
        super().__init__()

        self.heads = heads
        self.dropout = dropout

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self, x: Float[Tensor, "batch patch feature"]
    ) -> Float[Tensor, "batch patch feature"]:
        q, k, v = eo.rearrange(
            self.to_qkv(x), "b p (qkv h c) -> qkv b h p c", qkv=3, h=self.heads
        ).contiguous()

        dropout_p = self.dropout if self.training else 0.0
        out = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = eo.rearrange(out, "b h p c -> b p (h c)")
        return self.to_out(out)


def modulate(
    x: Float[Tensor, "batch patch feature"],
    shift: Float[Tensor, "batch feature"],
    scale: Float[Tensor, "batch feature"],
) -> Float[Tensor, "batch patch feature"]:
    return torch.addcmul(shift[:, None], scale[:, None] + 1, x)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(
        self, size: int, heads: int, mlp_ratio: int = 4, dropout: float | None = None
    ):
        super().__init__()

        self.norm = nn.LayerNorm(size, elementwise_affine=False)
        self.attn = Attention(
            size, heads=heads, dropout=dropout if dropout is not None else 0.0
        )
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.mlp = MLP(
            in_features=size,
            hidden_features=[mlp_ratio * size],
            out_features=size,
            actfn=partial(nn.GELU, approximate="tanh"),
        )
        # Added a linear layer in front of SiLU, so that not all layers just apply SiLU
        # to the exact same t embedding
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(size, size), nn.SiLU(), nn.Linear(size, 6 * size)
        )

        # Initialize DiT block as identity (-Zero initialization)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self, x: Float[Tensor, "batch patch feature"], c: Float[Tensor, "batch feature"]
    ) -> Float[Tensor, "batch patch feature"]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = torch.addcmul(
            x,
            gate_msa.unsqueeze(dim=1),
            self.attn(modulate(self.norm(x), shift_msa, scale_msa)),
        )
        x = torch.addcmul(
            x,
            gate_mlp.unsqueeze(dim=1),
            self.mlp(self.dropout(modulate(self.norm(x), shift_mlp, scale_mlp))),
        )
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        patch_size: int,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        depth: int,
        heads: int,
        mlp_ratio: int,
        dropout: float | None,
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        height, width = input_size
        patch_area = patch_size**2
        patches_h = height // patch_size
        patches_w = width // patch_size

        # Use fixed Fourier positional embeddings
        pos_embedding = NyquistPositionalEmbedding(hidden_size // 2, max(height, width))
        pos_h = pos_embedding(torch.linspace(0, 1, patches_h))
        pos_w = pos_embedding(torch.linspace(0, 1, patches_w))
        pos_embs = torch.cat(
            (
                eo.repeat(pos_h, "ph c -> (ph pw) c", pw=patches_w),
                eo.repeat(pos_w, "pw c -> (ph pw) c", ph=patches_h),
            ),
            dim=1,
        )
        self.register_buffer("patch_pos_embedding", pos_embs, persistent=False)
        self.t_embedding = NyquistPositionalEmbedding(hidden_size, 1000)

        self.patchify = Rearrange(
            "batch c (nh ps_h) (nw ps_w) -> batch (nh nw) (ps_h ps_w c)",
            ps_h=patch_size,
            ps_w=patch_size,
        )
        self.patch_encoder = nn.Linear(patch_area * in_channels, hidden_size)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )

        self.patch_decoder = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, patch_area * out_channels)
        )
        self.unpatchify = Rearrange(
            "batch (nh nw) (ps_h ps_w c) -> batch c (nh ps_h) (nw ps_w)",
            nh=height // patch_size,
            nw=width // patch_size,
            ps_h=patch_size,
            ps_w=patch_size,
        )

    def forward(
        self, x: Float[Tensor, "batch channel height width"], t: Float[Tensor, "batch"]
    ) -> Float[Tensor, "batch channel height width"]:
        c = self.t_embedding(t)
        x = self.patch_encoder(self.patchify(x)) + self.patch_pos_embedding
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.patch_decoder(x))


class DenoisingDiT(nn.Module):
    """Diffusion Transformer [1].

    [1] https://arxiv.org/abs/2212.09748
    """

    def __init__(
        self,
        data_shape: tuple[int, ...],
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        dropout: float | None = None,
        fourier_features: FourierFeatures | None = None,
        **kwargs,
    ):
        super().__init__()

        self.data_shape = tuple(data_shape)
        self.fourier_features = fourier_features

        assert len(self.data_shape) == 3, "Only works for 2D images"

        n_channels = data_shape[0]
        in_channels = out_channels = n_channels
        if fourier_features is not None:
            in_channels += n_channels * fourier_features.n_features()

        self.dit = DiT(
            input_size=data_shape[1:],
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=dim,
            depth=depth,
            heads=heads,
            mlp_ratio=4,
            dropout=dropout,
        )

    def forward(
        self, mu: Float[Tensor, "batch {self.data_shape}"], t: Float[Tensor, "batch"]
    ) -> Float[Tensor, "batch {self.data_shape}"]:
        x_parts = [mu]
        if self.fourier_features is not None:
            x_parts.append(self.fourier_features(mu, dim=1))
        x = torch.cat(x_parts, dim=1)

        return self.dit(x, t)
