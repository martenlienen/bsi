import random
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(config: DictConfig) -> tuple[np.random.Generator, np.random.SeedSequence]:
    """Set seed for random number generators in pytorch, numpy and python.random.

    The seed is a big int which can be a problem for some loggers, so we store it as a
    string instead.
    """

    big_seed = int(config.seed) if config.seed is not None else None
    big_seed, rng, seed_sequence = manual_seed(big_seed)
    config.seed = str(big_seed)
    return rng, seed_sequence


def manual_seed(seed: Optional[int]):
    """Seed all RNGs manually without reusing the same seed."""
    root_ss = np.random.SeedSequence(seed)

    num_rngs = 4
    if torch.cuda.is_available():
        num_rngs += torch.cuda.device_count()
    std_ss, np_ss, npg_ss, pt_ss, *cuda_ss = root_ss.spawn(num_rngs)

    # Python uses a Mersenne twister with 624 words of state, so we provide enough seed
    # to initialize it fully
    random.seed(std_ss.generate_state(624).tobytes())

    # It is best practice not to use numpy's global RNG, so we instantiate one
    rng = np.random.default_rng(np_ss)

    # We seed the global RNG anyway in case some library uses it internally
    np.random.seed(int(npg_ss.generate_state(1, np.uint32)))

    # Always initialize the global default (CPU) generator
    torch.random.default_generator.manual_seed(int(pt_ss.generate_state(1, np.uint64)))

    if torch.cuda.is_available():

        def lazy_seed_cuda():
            for i in range(torch.cuda.device_count()):
                device_seed = int(cuda_ss[i].generate_state(1, np.uint64))
                torch.cuda.default_generators[i].manual_seed(device_seed)

        torch.cuda._lazy_call(lazy_seed_cuda)

    seed = root_ss.entropy
    return seed, rng, root_ss


def torch_generator_seed(seed_sequence: np.random.SeedSequence) -> int:
    """Generate a seed for a torch generator."""

    return int(seed_sequence.spawn(1)[0].generate_state(1, np.uint64))
