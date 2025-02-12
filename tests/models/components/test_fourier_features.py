import einops as eo
import numpy as np
import torch
from numpy.testing import assert_allclose

from bsi.nn import FourierFeatures


def test_fourier_features():
    module = FourierFeatures(n_min=5, n_max=6)

    x = torch.tensor([1.333, -np.e / 7])
    x = eo.repeat(x, "c -> 2 c 3")
    y = module(x, dim=1)

    assert module.n_features() == 2 * 2
    assert y.shape == (2, 2 * 2 * 2, 3)
    expected = [
        np.sin(2 * np.pi * 2**5 * 1.333),
        np.cos(2 * np.pi * 2**5 * 1.333),
        np.sin(2 * np.pi * 2**6 * 1.333),
        np.cos(2 * np.pi * 2**6 * 1.333),
        np.sin(2 * np.pi * 2**5 * -np.e / 7),
        np.cos(2 * np.pi * 2**5 * -np.e / 7),
        np.sin(2 * np.pi * 2**6 * -np.e / 7),
        np.cos(2 * np.pi * 2**6 * -np.e / 7),
    ]
    assert_allclose(y[0, :, 0], expected)
