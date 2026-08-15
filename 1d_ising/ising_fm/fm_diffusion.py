import torch
import torch.nn as nn


class WhiteNoiseFlow1D(nn.Module):
    """
    White-noise flow matching path:

        x1 ~ N(0, I)
        x_t = (1 - t) x0 + (sigma_min + (1 - sigma_min)t) x1

    Conditional velocity:

        u_t(x_t | x0, x1) = d x_t / dt
                         = (1 - sigma_min) x1 - x0
    """

    def __init__(self, sigma_min=0.005):
        super().__init__()
        self.sigma_min = sigma_min

    @torch.no_grad()
    def sample_t(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    @torch.no_grad()
    def forward(self, x0, t=None):
        """
        x0: (B, 1, L)

        return:
            xt: (B, 1, L)
            ut: (B, 1, L)
            t:  (B,)
        """
        B = x0.shape[0]
        device = x0.device

        if t is None:
            t = self.sample_t(B, device)

        t_view = t.view(B, *([1] * (x0.ndim - 1)))

        x1 = torch.randn_like(x0)

        sm = self.sigma_min

        xt = (1.0 - t_view) * x0 + (sm + (1.0 - sm) * t_view) * x1

        ut = (1.0 - sm) * x1 - x0

        return xt, ut, t
