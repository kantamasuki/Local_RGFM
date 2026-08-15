import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / half
        )
        args = 2.0 * math.pi * t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb


class LocalConvDenoiser1D(nn.Module):
    """
    Fully convolutional local denoiser.

    input : x_t, shape (B, 1, L)
    time  : t,   shape (B,)
    output: eps, shape (B, 1, L)

    Receptive field:
        R = 1 + depth * (kernel_size - 1)
    """

    def __init__(self, hidden_dim=64, time_dim=64, depth=6, kernel_size=5):
        super().__init__()

        assert kernel_size % 2 == 1

        self.depth = depth
        self.kernel_size = kernel_size
        self.receptive_field = 1 + depth * (kernel_size - 1)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        padding = kernel_size // 2

        self.in_conv = nn.Conv1d(1, hidden_dim, kernel_size, padding=padding, padding_mode="reflect")

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, padding_mode="reflect"),
            )
            for _ in range(depth - 1)
        ])

        self.out = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, padding_mode="reflect"),
        )

    def forward(self, x, t):
        """
        x: (B, 1, L)
        t: (B,)
        """
        h = self.in_conv(x)

        temb = self.time_mlp(t)[:, :, None]  # (B, hidden_dim, 1)

        h = h + temb

        for block in self.blocks:
            h = h + block(h)

        return self.out(h)
