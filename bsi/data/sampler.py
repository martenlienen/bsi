import math

import torch
from torch import distributed
from torch.utils.data import Sampler


class InfiniteRandomSampler(Sampler):
    """Yields an endless stream of random samples from a data source.

    This is a distributed sampler, so indices are distributed across workers.
    """

    def __init__(self, data_source, *, generator: torch.Generator | None = None):
        super().__init__()

        self.data_source = data_source
        self.generator = generator

        if distributed.is_initialized():
            self.world_size = distributed.get_world_size()
            self.rank = distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def __len__(self):
        return math.inf

    def __iter__(self):
        n = len(self.data_source)
        # We yield every `world_size`-th element of an infinite list of permutations of
        # the dataset indices with an initial offset of `rank`.
        world_size = self.world_size
        offset = self.rank
        idxs = torch.randperm(n, generator=self.generator)
        while True:
            if offset >= n:
                idxs = torch.randperm(n, generator=self.generator)
                offset = offset % n
            yield idxs[offset]
            offset = offset + world_size


class DistributedNonPaddingSampler(Sampler):
    """Similar to distributed sampler but does not pad batches with extra samples.

    This is important for an accurate evaluation.
    """

    def __init__(self, data_source):
        super().__init__()

        self.data_source = data_source

        if distributed.is_initialized():
            self.world_size = distributed.get_world_size()
            self.rank = distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.sample_idxs = list(range(self.rank, len(self.data_source), self.world_size))

    def __len__(self):
        return len(self.sample_idxs)

    def __iter__(self):
        yield from self.sample_idxs
