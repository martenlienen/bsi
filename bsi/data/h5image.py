import logging
from pathlib import Path

import h5py as h5
import numpy as np
import torch
from loky import get_reusable_executor
from torch import Tensor, distributed
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


def _load_samples_worker(dataset: Dataset, start: int, end: int):
    samples = [dataset[i] for i in range(start, end)]
    images, labels = list(map(list, zip(*samples)))

    return np.array(images), np.array(labels)


class H5FileReader:
    @staticmethod
    def _read_dataset_shape(path: Path) -> tuple[int, ...]:
        with h5.File(path, mode="r") as f:
            assert "data" in f, f"No 'data' field found in {path}"
            assert "label" in f, f"No 'label' field found in {path}"
            return f["data"].shape

    def __init__(self, path: Path):
        super().__init__()

        self.path = path

        self.n, *self.data_shape = self._read_dataset_shape(path)

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.path}, n={self.n}, data_shape={self.data_shape})"

    def read(self, index: int | list[int] | slice):
        if isinstance(index, list):
            # h5py requires indices to be sorted and unique when reading
            idxs = np.asarray(index)
            unique_sorted_idxs, inverse_idx = np.unique(idxs, return_inverse=True)

        container = {}

        def read_idxs(name: str, value: h5.Dataset):
            if isinstance(index, list):
                sorted_values = np.array(value[unique_sorted_idxs])
                # Undo the sorting and uniquification
                index_values = sorted_values[inverse_idx]
            else:
                index_values = np.array(value[index])

            container[name] = torch.as_tensor(index_values)

        with h5.File(self.path, mode="r") as f:
            f.visititems(read_idxs)

        return torch.as_tensor(container["data"]), torch.as_tensor(container["label"])


def slice_to_indices(s: slice, x: np.ndarray) -> np.ndarray:
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else len(x)
    step = s.step if s.step is not None else 1
    return np.arange(start, stop, step)


class OnDemandH5InMemoryCache:
    def __init__(self, reader: H5FileReader, device: torch.device | None = None):
        super().__init__()

        self.reader = reader
        self.n = reader.n
        self.data_shape = reader.data_shape

        self.must_load = np.ones((self.n,), dtype=bool)
        self.data = torch.empty(
            (self.n, *self.data_shape), device=device, dtype=torch.float32
        )
        self.labels = torch.empty((self.n,), device=device, dtype=torch.long)

    def __repr__(self):
        return f"{self.__class__.__name__}(reader={self.reader!r})"

    def read(self, index: int | list[int] | slice):
        if isinstance(index, list):
            # Indexing tensors with an array is significantly faster than with a list,
            # even if you take the conversion time into account. numpy is just a lot
            # faster than pytorch at converting from lists.
            index = np.array(index)

        ########################################
        # Read any missing samples into memory #
        ########################################

        requested = index
        if isinstance(requested, slice):
            requested = slice_to_indices(requested, self.must_load)
        elif isinstance(requested, int):
            requested = np.array([requested])
        to_load = self.must_load[requested]
        if np.any(to_load):
            load_idxs = requested[to_load]
            self.must_load[load_idxs] = False
            load_idxs_torch = torch.as_tensor(load_idxs, device=self.data.device)
            data, labels = self.reader.read(load_idxs.tolist())
            self.data[load_idxs_torch] = data.to(device=self.data.device)
            self.labels[load_idxs_torch] = labels.to(device=self.labels.device)

        ###################
        # Read from cache #
        ###################

        return self.data[index], self.labels[index]


class PreloadedH5InMemoryCache:
    def __init__(self, reader: H5FileReader, device: torch.device | None = None):
        super().__init__()

        self.reader = reader
        self.n = reader.n
        self.device = device

        self.data = None
        self.labels = None

    def __repr__(self):
        return f"{self.__class__.__name__}(reader={self.reader!r})"

    def read(self, index: int | list[int] | slice):
        if self.data is None:
            # Cache data on first read to avoid delays in DDP when the rank-0 process is
            # responsible for starting the other processes
            data, labels = self.reader.read(slice(None, None))
            self.data = data.to(device=self.device, dtype=torch.float32)
            self.labels = labels.to(device=self.device, dtype=torch.long)

        return self.data[index], self.labels[index]


class H5ImageDataset(Dataset):
    """A labeled image dataset cached in an HDF5 file."""

    @staticmethod
    def create_from_dataset(
        path: Path, dataset: Dataset, *, name: str = "dataset", block_size: int = 512
    ):
        """Cache a labeled image dataset like CIFAR10 in an HDF5 file"""

        n = len(dataset)
        assert n > 0, "Cannot cache empty datasets"

        if path.is_file():
            n_cached, *_ = H5FileReader._read_dataset_shape(path)
            assert n == n_cached
            return

        path.parent.mkdir(exist_ok=True, parents=True)
        with h5.File(path, mode="w") as f:
            image, label = dataset[0]
            image = np.array(image)
            f.create_dataset("data", shape=(n, *image.shape), dtype=image.dtype)
            f.create_dataset("label", shape=(n,), dtype=np.array(label).dtype)

            block_starts = list(range(0, n, block_size))
            block_ends = [min(b + block_size, n) for b in block_starts]
            pool = get_reusable_executor()
            jobs = pool.map(
                _load_samples_worker,
                [dataset] * len(block_starts),
                block_starts,
                block_ends,
            )
            for start, end, (images, labels) in tqdm(
                zip(block_starts, block_ends, jobs),
                total=len(block_starts),
                desc=f"Caching {name}",
            ):
                f["data"][start:end] = images
                f["label"][start:end] = labels

    @staticmethod
    def create_from_data(
        path: Path, data: np.ndarray, labels: np.ndarray, *, name: str = "dataset"
    ):
        """Cache a labeled image dataset like MNIST in an HDF5 file"""

        n = len(data)
        assert n > 0, "Cannot cache empty datasets"

        if path.is_file():
            n_cached, *_ = H5FileReader._read_dataset_shape(path)
            assert n == n_cached
            return

        path.parent.mkdir(exist_ok=True, parents=True)
        with h5.File(path, mode="w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("label", data=labels)

    def __init__(
        self,
        path: Path,
        *,
        in_memory: bool = False,
        preload: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.path = path
        self.in_memory = in_memory
        self.preload = preload
        self.device = device

        self.reader = H5FileReader(self.path)
        if self.in_memory:
            if self.preload:
                self.reader = PreloadedH5InMemoryCache(self.reader, device=device)
            else:
                self.reader = OnDemandH5InMemoryCache(self.reader, device=device)

    def __len__(self):
        return self.reader.n

    def __getitem__(self, index: int | list[int] | slice) -> tuple[Tensor, Tensor]:
        return self.reader.read(index)

    def __getitems__(self, indices: list[int] | slice) -> tuple[Tensor, Tensor]:
        """Read multiple items at once."""

        # Signals to the data loader that this dataset supports reading multiple indices
        # at once.

        return self.__getitem__(indices)


class H5DataMixin:
    def __init__(
        self,
        *,
        num_workers: int = 0,
        pin_memory: bool | None = None,
        in_memory: bool = False,
        preload: bool = True,
        on_device: bool = False,
        persistent_workers: bool = True,
        **kwargs,
    ):
        self.in_memory = in_memory
        self.preload = preload
        self.on_device = on_device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        if self.pin_memory is None:
            self.pin_memory = self.num_workers > 0

        if self.pin_memory and self.num_workers == 0:
            log.warning(
                "Memory pinning without data workers creates unnecessary copies. "
                "Disabling it."
            )
            self.pin_memory = False

        if self.on_device:
            if torch.cuda.is_available():
                if torch.cuda.device_count() == 1:
                    self.device = torch.device("cuda")

                    if self.pin_memory:
                        log.warning(
                            "Memory pinning does not make sense when the data is cached "
                            "on-device. Disabling it."
                        )
                        self.pin_memory = False
                else:
                    log.warning(
                        "On-device caching requested but training uses multiple "
                        "GPUs, so caching on one of them does not work."
                    )
                    self.device = None
            else:
                log.warning("On-device caching requested but no GPUs are available.")
                self.device = None
        else:
            self.device = None

    def _h5_dataset(self, path: Path):
        return H5ImageDataset(
            path, in_memory=self.in_memory, preload=self.preload, device=self.device
        )

    def _data_loader(
        self,
        dataset,
        *,
        batch_size: int,
        sampler=None,
        allow_persistent_workers: bool = True,
    ):
        # Split the batch across devices
        if distributed.is_initialized():
            world_size = distributed.get_world_size()
            rank = distributed.get_rank()
            batch_size = batch_size // world_size + int(rank < (batch_size % world_size))

        return StatefulDataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda batch: batch,
            persistent_workers=allow_persistent_workers
            and (self.num_workers > 0 and self.persistent_workers),
        )
