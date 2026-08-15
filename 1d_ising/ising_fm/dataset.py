import torch
from torch.utils.data import Dataset
import numpy as np

class IsingNpyDataset(Dataset):
    """
    Dataset for saved 1D Ising samples.

    Expected .npy shape:
        (N, L)       values in {-1, +1}, or optionally {0, 1}
        (N, 1, L)    also accepted

    Returned tensor:
        float32, shape (1, L)
    """

    def __init__(
        self,
        data_path,
        L=1024,
        max_samples=None,
        normalize=False,
        subtract_mean=False,
        map_01_to_pm1=True,
        mmap_mode="r",
    ):
        if data_path is None or str(data_path) == "":
            raise ValueError("For dataset_key='ising', config must contain a non-empty data_path.")

        self.data_path = str(data_path)
        self.L = int(L)
        self.normalize = bool(normalize)
        self.subtract_mean = bool(subtract_mean)
        self.map_01_to_pm1 = bool(map_01_to_pm1)

        arr = np.load(self.data_path, mmap_mode=mmap_mode)
        if arr.ndim == 2:
            if arr.shape[1] != self.L:
                raise ValueError(f"Expected data shape (N,{self.L}), got {arr.shape}.")
        elif arr.ndim == 3:
            if arr.shape[1] != 1 or arr.shape[2] != self.L:
                raise ValueError(f"Expected data shape (N,1,{self.L}), got {arr.shape}.")
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}.")

        if max_samples is not None:
            max_samples = int(max_samples)
            arr = arr[:max_samples]

        self.data = arr

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        x = np.asarray(self.data[idx])

        if x.ndim == 2:
            # (1, L) -> (L,)
            x = x[0]

        x = torch.from_numpy(x.astype(np.float32, copy=False))

        # Accept {0,1} files as well as {-1,+1} files.
        if self.map_01_to_pm1:
            xmin = float(x.min())
            xmax = float(x.max())
            if xmin >= -1e-6 and xmax <= 1.0 + 1e-6:
                x = 2.0 * x - 1.0

        if self.subtract_mean:
            x = x - x.mean()

        if self.normalize:
            x = x / (x.std() + 1e-6)

        return x.unsqueeze(0)  # shape: (1, L)


def get_dataset(
      dataset_key,
      L=1024,
      data_path=None,
      max_samples=None,
      normalize=False,
      subtract_mean=False,
      mmap_mode="r",
      ):
    if dataset_key == "ising":
        return IsingNpyDataset(
            data_path=data_path,
            L=L,
            max_samples=max_samples,
            normalize=normalize,
            subtract_mean=subtract_mean,
            mmap_mode=mmap_mode,
        )
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")
