import math
import torch
from torch.utils.data import Dataset


class LowFreqSineDataset(Dataset):
    """
    1D toy dataset:
        x_i = sum_q A_q cos(2pi q i/L + theta_q) + small white noise

    Each sample has coherent low-frequency global structure.
    Local patch-only diffusion should have difficulty maintaining global phase coherence.
    """

    def __init__(
        self,
        num_samples=100_000,
        L=1024,
        modes=(1, 2, 3),
        amp_std=1.0,
        obs_noise=0.05,
        normalize=False,
    ):
        self.num_samples = num_samples
        self.L = L
        self.modes = modes
        self.amp_std = amp_std
        self.obs_noise = obs_noise
        self.normalize = normalize

        self.grid = torch.arange(L).float() / L

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.zeros(self.L)

        for q in self.modes:
            amp = self.amp_std * torch.randn(())
            phase = 2 * math.pi * torch.rand(())
            x += amp * torch.cos(2 * math.pi * q * self.grid + phase)

        x += self.obs_noise * torch.randn_like(x)
        
        if self.normalize:
            x = x - x.mean()
            x = x / (x.std() + 1e-6)
        else:
            x = x - x.mean()

        return x.unsqueeze(0)  # shape: (1, L)


def get_dataset(dataset_key, L=1024):
    if dataset_key == "lowfreq_sine":
        return LowFreqSineDataset(
            num_samples=100_000,
            L=L,
            modes=(1, 2, 3),
            amp_std=1.0,
            obs_noise=0.05,
            normalize=False,
        )
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")
