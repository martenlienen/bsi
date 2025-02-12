from pathlib import Path
from typing import Literal

import einops as eo
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import Subset, random_split
from tqdm import tqdm

from ..utils import relative_to_project_root
from .h5image import H5DataMixin, H5ImageDataset
from .sampler import DistributedNonPaddingSampler, InfiniteRandomSampler


def read_imagenet_n(root: Path, *, split: Literal["train", "val"], n: int):
    data, labels = [], []
    for f in tqdm(
        sorted(root.glob(f"**/{split}_*.npz"), key=lambda p: p.name),
        desc=f"ImageNet{n}",
    ):
        part = np.load(f)
        data.append(part["data"])
        labels.append(part["labels"])
    data = eo.rearrange(np.concatenate(data), "b (c w h) -> b c w h", w=n, h=n)
    labels = np.concatenate(labels)

    return data, labels


class ImageNetDataModule(H5DataMixin, pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        *,
        n: int,
        batch_size: int,
        eval_batch_size: int | None,
        **kwargs,
    ):
        H5DataMixin.__init__(self, **kwargs)
        pl.LightningDataModule.__init__(self)

        self.root = relative_to_project_root(root)
        self.n = n
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.seed = 14196134745989613036

        self.train_data = None

    def prepare_data(self):
        def transform(data: np.ndarray):
            # pytorch's transforms are not vectorized and preparing the data takes
            # hours, when it could be done in a few minutes.
            return torch.as_tensor(data, dtype=torch.float32) * (2 / 255) - 1

        train_path = self._cache_path("train")
        if not train_path.is_file():
            data, labels = read_imagenet_n(self.root / "data", split="train", n=self.n)
            H5ImageDataset.create_from_data(
                train_path, transform(data), labels, name=self.short_name()
            )
        test_path = self._cache_path("test")
        if not test_path.is_file():
            data, labels = read_imagenet_n(self.root / "data", split="val", n=self.n)
            H5ImageDataset.create_from_data(
                test_path, transform(data), labels, name=self.short_name()
            )

    def _cache_path(self, stage: str):
        return self.root / f"{stage}.h5"

    def setup(self, stage):
        if self.train_data is None:
            self.train_data = self._h5_dataset(self._cache_path("train"))

        # Subset of training data to evaluate on
        train_eval_subset_seed = 5308798251198469321
        eval_split_gen = torch.Generator().manual_seed(train_eval_subset_seed)
        train_eval_split_idx = torch.randperm(
            len(self.train_data), generator=eval_split_gen
        )

        if stage in ("fit", "validate"):
            self.val_train_split = Subset(
                self.train_data, train_eval_split_idx[:5_000].tolist()
            )

            train_val_split_seed = 11893635380066140590
            split_generator = torch.Generator().manual_seed(train_val_split_seed)
            val_len = int(0.01 * len(self.train_data))
            self.train_split, self.val_split = random_split(
                self.train_data,
                lengths=(len(self.train_data) - val_len, val_len),
                generator=split_generator,
            )
        elif stage == "test":
            self.test_data = self._h5_dataset(self._cache_path("test"))
            # Evaluate on a random subset of training data
            self.test_train_split = Subset(
                self.train_data, train_eval_split_idx[: len(self.test_data)].tolist()
            )

    def fid_train_dataloader(self):
        return self._data_loader(self.train_data, batch_size=self.batch_size)

    def train_dataloader(self):
        gen = torch.Generator().manual_seed(self.seed)
        return self._data_loader(
            self.train_split,
            batch_size=self.batch_size,
            sampler=InfiniteRandomSampler(self.train_split, generator=gen),
        )

    def val_dataloader(self):
        batch_size = self.eval_batch_size or len(self.val_split)
        val_loader = self._data_loader(
            self.val_split,
            batch_size=batch_size,
            sampler=DistributedNonPaddingSampler(self.val_split),
        )
        train_loader = self._data_loader(
            self.val_train_split,
            batch_size=batch_size,
            sampler=DistributedNonPaddingSampler(self.val_train_split),
        )
        return [val_loader, train_loader]

    def test_dataloader(self):
        batch_size = self.eval_batch_size or len(self.test_data)
        test_loader = self._data_loader(
            self.test_data,
            batch_size=batch_size,
            # Only runs once, so we don't need to keep the workers around
            allow_persistent_workers=False,
            sampler=DistributedNonPaddingSampler(self.test_data),
        )
        train_loader = self._data_loader(
            self.test_train_split,
            batch_size=batch_size,
            # Only runs once, so we don't need to keep the workers around
            allow_persistent_workers=False,
            sampler=DistributedNonPaddingSampler(self.test_train_split),
        )
        return [test_loader, train_loader]

    def data_shape(self) -> tuple[int, ...]:
        return (3, self.n, self.n)

    def short_name(self) -> str:
        return f"imagenet{self.n}"
