import torch
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    """Compute Fourier features as proposed in the VDM paper [1].

    [1] https://arxiv.org/abs/2006.10739
    """

    def __init__(self, *, n_min: int, n_max: int, **kwargs):
        super().__init__()

        self.n_min = n_min
        self.n_max = n_max

        ns = torch.arange(n_min, n_max + 1)
        self.register_buffer("coefs", 2 * torch.pi * 2**ns, persistent=False)
        self.register_buffer("offsets", torch.tensor([0, torch.pi / 2]), persistent=False)

    def n_features(self):
        return len(self.coefs) * len(self.offsets)

    def forward(self, x: Tensor, *, dim: int) -> Tensor:
        assert dim >= 0, "Implementation expects a non-negative dimension index"

        # Multiply each feature with each frequency and add each offset by expanding the
        # channel dimension twice and then aligning the cofficient and offset tensors
        # with them.
        right_dims = x.dim() - dim - 1
        x = x.unsqueeze(dim + 1).unsqueeze(dim + 1)
        coefs = self.coefs.view((-1, *([1] * (right_dims + 1))))
        offsets = self.offsets.view((-1, *([1] * right_dims)))
        args = torch.addcmul(offsets, coefs, x)

        return args.sin().flatten(start_dim=dim, end_dim=dim + 2)
