from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torchmetrics as tm
from tqdm import tqdm


@contextmanager
def num_torch_threads(n: int):
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(n)
        yield
    finally:
        torch.set_num_threads(old_threads)


class FIDScore(tm.image.FrechetInceptionDistance):
    """
    A wrapper around torchmetrics FID that keeps the FID network out of checkpoints and
    uses precomputed dataset statistics instead of recomputing them for every
    evaluation.
    """

    def __init__(self, *, feature: int, stats_path: Path):
        super().__init__(feature=feature, reset_real_features=True, normalize=True)

        self.stats_path = stats_path
        self.block_size = 256

        self.register_load_state_dict_post_hook(self.filter_missing_fid_state)

        stats = np.load(stats_path)
        self.register_buffer(
            "real_n", torch.as_tensor(stats["n"], dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "real_sum",
            torch.as_tensor(stats["sum"], dtype=torch.double),
            persistent=False,
        )
        self.register_buffer(
            "real_cov_sum",
            torch.as_tensor(stats["cov_sum"], dtype=torch.double),
            persistent=False,
        )

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # Keep the inception network out of the state dict and checkpoints
        for key in list(state.keys()):
            if "inception." in key:
                del state[key]
        return state

    def filter_missing_fid_state(self, module, incompatible_keys):
        # Ignore the "missing" state of the FID network that we filtered in #state_dict,
        # so that strict checkpoint loading still works. incompatible_keys is meant to
        # be modified in-place.
        for i in reversed(range(len(incompatible_keys.missing_keys))):
            if "inception." in incompatible_keys.missing_keys[i]:
                del incompatible_keys.missing_keys[i]

    def update(self, samples):
        # The samples are upscaled to compute the embeddings and this can take a lot of
        # memory, so we split the samples into blocks.
        for i in tqdm(range(0, len(samples), self.block_size), desc="FID", leave=False):
            super().update(samples[i : i + self.block_size], real=False)

    def compute(self):
        # Reset real sample stats for every computation to ensure that they are not
        # screwed up by DDP synchronization
        self.real_features_num_samples = self.real_n.clone()
        self.real_features_sum = self.real_sum.clone()
        self.real_features_cov_sum = self.real_cov_sum.clone()

        # On our cluster, torch.linalg.eigvals runtime scales badly with the number of
        # CPU threads and torch.linalg.eigvals always computes on the CPU even for GPU
        # tensors. This makes the FID computation take up to a minute. By restricting
        # the number of threads, the time for the FID becomes a reasonable ~3 seconds.
        with num_torch_threads(1):
            return super().compute()
