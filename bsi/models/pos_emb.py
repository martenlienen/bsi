import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn


class NyquistPositionalEmbedding(nn.Module):
    """Sine-cosine embedding for position `t` in `[0, 1]` that scales from a frequency
    of 1/8 to a (< 1) multiple of the Nyquist frequency.

    In the original sinusoidal embedding from "Attention is all you need" [1], the high
    frequencies are so high that the corresponding embedding features are basically
    random and the inner product between two embeddings `e(t_0)` and `e(t_1)` becomes
    noisy instead of varying smoothly as `t_0 - t_1` grows. At the same time, the
    slowest frequency is still so high that no feature just varies linearly with `t`,
    which we believe to be important.

    We choose 1/8 as the slowest frequency so that the slowest-varying embedding varies
    roughly lineary across [0, 2pi] as the relative error between x and sin(x) on [0,
    2pi / 8] is at most 2.5%. For the highest frequency, we pick a value dependent on
    the Nyquist frequency of the expected sampling rate of the embedding.

    The Nyquist frequency is the largest frequency that one can sample at a given rate
    without aliasing, so one could assume that to be a great choice for the highest
    frequency but sampling sine and cosine at the Nyquist frequency would result in
    constant (and therefore uninformative) 1 and 0 features.

    Instead, we choose the maximum frequency as Nyquist/(2*phi) where phi is the golden
    ratio. The exact choice is arbitrary as long as it is some irrational number
    somewhat smaller than the Nyquist frequency. The irrationality ensures that the
    high-frequency features do not oscillate between a fixed number of values, e.g. at
    Nyquist/2, the fastest-varying feature would just take the values -1, 0 and 1 when
    sampled at the given expected rate.

    [1] https://arxiv.org/abs/1706.03762
    """

    @classmethod
    def from_config(cls, size, expected_rate, **kwargs):
        return cls(size, expected_rate)

    def __init__(self, size: int, expected_rate: int):
        """Construct the embedding.

        Args:
            size: Embedding dimensions.
            expected_rate: Expected sampling rate to compute the Nyquist frequency and
              thus the frequencies for the embeddings. A value of `10` means that you
              expect embeddings to be computed for 1/10, 2/10, ..., 1.
        """

        super().__init__()

        self.size = size

        assert size % 2 == 0

        k = size // 2

        # Nyquist frequency for a given sampling rate per unit interval
        nyquist_frequency = expected_rate / 2

        golden_ratio = (1 + np.sqrt(5)) / 2
        frequencies = np.geomspace(1 / 8, nyquist_frequency / (2 * golden_ratio), num=k)

        # Sample every frequency twice, once shifted by pi/2 to get cosine
        scale = np.repeat(2 * np.pi * frequencies, 2)
        bias = np.tile(np.array([0, np.pi / 2]), k)

        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "bias", torch.tensor(bias, dtype=torch.float32), persistent=False
        )

    def forward(self, t: Float[Tensor, "..."]) -> Float[Tensor, "... size"]:
        """Embed the timestep t into a sine-cosine representation.

        Args:
            t: Batch of timesteps between 0 and 1.
        """

        return torch.addcmul(self.bias, self.scale, t[..., None]).sin()
