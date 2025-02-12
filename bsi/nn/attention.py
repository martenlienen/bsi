import einops as eo
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend


def fused_attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    """Scaled dot product attention with one of the fused, memory efficient kernels."""

    with torch.nn.attention.sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        ]
    ):
        return F.scaled_dot_product_attention(q, k, v, **kwargs)


class Attention2D(nn.Module):
    """Attention over a 2D image."""

    def __init__(self, dim: int, *, heads: int = 4, padding_mode: str = "zeros"):
        super().__init__()

        self.heads = heads

        self.to_qkv = nn.Conv2d(dim, dim * 3, 3, padding=1, padding_mode=padding_mode)
        self.to_out = nn.Conv2d(dim, dim, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = eo.rearrange(
            self.to_qkv(x), "b (qkv h c) x y -> qkv b h (x y) c", qkv=3, h=self.heads
        ).contiguous()

        out = fused_attention(q, k, v)

        out = eo.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)
