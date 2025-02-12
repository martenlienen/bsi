import torch
from numpy.testing import assert_allclose

from bsi.bsi import Discretization


def test_bucketize_rgb():
    disc = Discretization(0.0, 1.0, k=256)

    x = torch.tensor([-0.1, 0.0, 1.0, 1.0 - 1 / 256])
    idx = disc.bucketize(x)

    assert_allclose(idx, [0, 0, 255, 254])


def test_bucketize_aligns_with_boundaries():
    disc = Discretization(-1.0, 1.0, k=5)
    boundaries = disc.bin_boundaries(torch.device("cpu"), torch.float64)

    idx = disc.bucketize(boundaries)[:-1]
    assert_allclose(idx, list(range(5)))

    idx = disc.bucketize(boundaries - 1e-8)[1:]
    assert_allclose(idx, list(range(5)))


def test_bin_boundaries():
    disc = Discretization(-1.0, 1.0, k=3)

    # Bin centers are at -1, 0, 1
    assert_allclose(
        disc.bin_boundaries(torch.device("cpu"), torch.float32),
        [-3 / 2, -1 / 2, 1 / 2, 3 / 2],
    )
