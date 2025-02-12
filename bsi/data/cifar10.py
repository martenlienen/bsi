import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from ..utils import relative_to_project_root
from .h5image import H5DataMixin, H5ImageDataset
from .sampler import DistributedNonPaddingSampler, InfiniteRandomSampler


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: v2.Transform):
        super().__init__()

        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return self.transform(data), label

    def __getitems__(self, idxs):
        data, label = self.dataset.__getitems__(idxs)
        return self.transform(data), label


class CIFAR10DataModule(H5DataMixin, pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        *,
        batch_size: int,
        eval_batch_size: int | None,
        augment: dict,
        **kwargs,
    ):
        H5DataMixin.__init__(self, **kwargs)
        pl.LightningDataModule.__init__(self)

        self.root = relative_to_project_root(root)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.augment = augment
        self.seed = 1731901944267979080

        self.train_data = None

    def prepare_data(self):
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        train_data = CIFAR10(self.root, train=True, download=True, transform=transforms)
        H5ImageDataset.create_from_dataset(
            self.root / "train.h5", train_data, name="CIFAR10"
        )
        test_data = CIFAR10(self.root, train=False, download=True, transform=transforms)
        H5ImageDataset.create_from_dataset(
            self.root / "test.h5", test_data, name="CIFAR10"
        )

    def setup(self, stage):
        if self.train_data is None:
            self.train_data = self._h5_dataset(self.root / "train.h5")

        # Subset of training data to evaluate on
        train_eval_subset_seed = 8288933137687132059
        eval_split_gen = torch.Generator().manual_seed(train_eval_subset_seed)
        train_eval_split_idx = torch.randperm(
            len(self.train_data), generator=eval_split_gen
        )

        if stage in ("fit", "validate"):
            self.val_train_split = Subset(
                self.train_data, train_eval_split_idx[:5_000].tolist()
            )

            train_val_split_seed = 11812925458092569678
            split_generator = torch.Generator().manual_seed(train_val_split_seed)
            self.train_split, self.val_split = random_split(
                self.train_data, lengths=(0.9, 0.1), generator=split_generator
            )

            transforms = []
            if self.augment.get("flip", False):
                transforms.append(v2.RandomHorizontalFlip())
            if len(transforms) > 0:
                random_transform = v2.Compose(transforms)
                self.train_split = TransformedDataset(self.train_split, random_transform)
        elif stage == "test":
            self.test_data = self._h5_dataset(self.root / "test.h5")
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
        return (3, 32, 32)

    def short_name(self) -> str:
        return "cifar10"
