# fm_diffusion.py

import torch
import torch.nn as nn


class FMDiffusion(nn.Module):
    """
    Standard flow matching path.

    x_0 ~ p_data
    x_1 ~ N(0, I)

    x_t = (1 - t) x_0 + (sigma_min + t(1 - sigma_min)) x_1

    u_t = d x_t / dt
        = -x_0 + (1 - sigma_min) x_1
    """

    def __init__(self, sigma_min=0.01):
        super().__init__()
        self.sigma_min = sigma_min

    @torch.no_grad()
    def forward(self, x0, t):
        """
        x0: [B, 3, L, L]
        t:  [B]

        return:
            x_t: [B, 3, L, L]
            u_t: [B, 3, L, L]
        """
        B = x0.shape[0]
        sm = self.sigma_min

        t_view = t.view(B, *([1] * (x0.ndim - 1)))

        x1 = torch.randn_like(x0)

        x_t = (1.0 - t_view) * x0 + (sm + (1.0 - sm) * t_view) * x1
        u_t = (1.0 - sm) * x1 - x0

        return x_t, u_t

    @torch.no_grad()
    def sample_prior(self, batch_size, channels, img_size, device):
        """
        Sample x_1 ~ N(0,I).
        """
        return torch.randn(batch_size, channels, img_size, img_size, device=device)
