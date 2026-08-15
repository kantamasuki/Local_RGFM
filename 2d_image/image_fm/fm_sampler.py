# fm_sampler.py

import torch
import torch.nn as nn
from torchdiffeq import odeint


class FMSamplerReal(nn.Module):
    """
    Standard flow matching sampler with local patch model.

    model:
        input : [B*N, 5, P, P]
        output: [B*N, 3, P, P]

    Sampling:
        x_1 ~ N(0,I)
        dx_t/dt = v_theta(x_t,t)
        integrate t=1 -> 0
    """

    def __init__(
        self,
        model,
        patch_maker,
        img_size,
        n_steps=300,
        method="midpoint",
        rtol=1e-5,
        atol=1e-5,
    ):
        super().__init__()

        self.model = model
        self.patch_maker = patch_maker
        self.L = img_size
        self.n_steps = n_steps
        self.method = method
        self.rtol = rtol
        self.atol = atol

        assert n_steps > 0

    def ode_func(self, t, x):
        """
        t: scalar tensor
        x: [B, 3, L, L]
        """
        B = x.shape[0]

        t_batch = t.expand(B)

        inp = self.patch_maker.make_input_batch(x)  # [B*N, 5, P, P]
        t_patch = t_batch.repeat_interleave(self.patch_maker.n_patch)

        pred_patch = self.model(inp, t_patch)       # [B*N, 3, P, P]

        u = self.patch_maker.reconstruct_from_pred_batch_simple(
            pred_patch,
            batch_size=B,
        )

        return u

    @torch.no_grad()
    def sample_prior(self, batch_size, channels=3, device=None):
        if device is None:
            device = next(self.model.parameters()).device

        return torch.randn(
            batch_size,
            channels,
            self.L,
            self.L,
            device=device,
        )

    @torch.no_grad()
    def forward(self, batch_size, channels=3, device=None, return_traj=False):
        """
        Sample from standard Gaussian prior.
        """
        if device is None:
            device = next(self.model.parameters()).device

        x1 = self.sample_prior(
            batch_size=batch_size,
            channels=channels,
            device=device,
        )

        return self.solve(x1, t_start=1.0, return_traj=return_traj)

    @torch.no_grad()
    def solve(self, x_start, t_start=1.0, return_traj=False):
        """
        Integrate from t_start to 0.

        x_start: [B, 3, L, L]
        """
        device = x_start.device

        t_span = torch.tensor([float(t_start), 0.0], device=device)

        options = None
        if self.method in ["euler", "midpoint", "rk4"]:
            options = {"step_size": float(t_start) / self.n_steps}

        x_traj = odeint(
            self.ode_func,
            x_start,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options,
        )

        if return_traj:
            return x_traj

        return x_traj[-1]


class FMReconstructor(nn.Module):
    """
    For monitoring.

    Given data x0, make x_t by FMDiffusion, then integrate back to t=0.
    """

    def __init__(self, fm_diffusion, fm_sampler):
        super().__init__()

        self.fm_diffusion = fm_diffusion
        self.fm_sampler = fm_sampler

    @torch.no_grad()
    def forward(self, x0, t_start=1.0, return_noisy=False):
        """
        x0: [B, 3, L, L]
        """
        B = x0.shape[0]
        device = x0.device

        t = torch.full((B,), float(t_start), device=device)
        x_t, _ = self.fm_diffusion(x0, t)

        x_rec = self.fm_sampler.solve(
            x_t,
            t_start=t_start,
            return_traj=False,
        )

        if return_noisy:
            return x_rec, x_t

        return x_rec
