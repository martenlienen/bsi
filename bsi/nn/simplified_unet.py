import torch
from jaxtyping import Float
from torch import Tensor, nn


class SimplifiedUNet(nn.Module):
    """A simplified U-Net structure without downsampling."""

    def __init__(
        self,
        downsampling_blocks: list[nn.Module | list[nn.Module]],
        upsampling_blocks: list[nn.Module | list[nn.Module]],
        center_block: nn.Module,
    ):
        super().__init__()

        assert len(downsampling_blocks) == len(upsampling_blocks)

        self.downsampling_blocks = nn.ModuleList(
            [
                nn.ModuleList(blocks if isinstance(blocks, list) else [blocks])
                for blocks in downsampling_blocks
            ]
        )
        self.upsampling_blocks = nn.ModuleList(
            [
                nn.ModuleList(blocks if isinstance(blocks, list) else [blocks])
                for blocks in upsampling_blocks
            ]
        )
        self.center_block = center_block

    def forward(self, x: Float[Tensor, "batch channel *dims"], *args, **kwargs):
        skips = []

        for level_blocks in self.downsampling_blocks:
            for block in level_blocks:
                x = block(x, *args, **kwargs)
                skips.append(x)

        x = self.center_block(x, *args, **kwargs)

        for level_blocks in self.upsampling_blocks:
            for block in level_blocks:
                x_skip = skips.pop()
                x = block(torch.cat((x, x_skip), dim=-3), *args, **kwargs)

        return x
