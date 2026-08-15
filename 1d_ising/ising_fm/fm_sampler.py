# fm_sampler.py

import torch
import torch.nn as nn
from torchdiffeq import odeint


class FMSamplerReal(nn.Module):
    """
    Real-space local-patch sampler with chain size L.

    Integrates from t=1 to t=0 using torchdiffeq.odeint.

    model:
        input : [B, 1, L]
        output: [B, 1, L]
    """

    def __init__(
        self,
        model,
        n_total_steps=200,
        method="midpoint",   # RK2
        rtol=1e-5,
        atol=1e-5,
    ):
        super().__init__()

        self.model = model
        self.n_total_steps = n_total_steps
        self.method = method
        self.rtol = rtol
        self.atol = atol

        assert n_total_steps > 0

    def ode_func(self, t, x):
        """
        t: scalar tensor
        x: [B, 1, L]

        return:
            u: [B, 1, L]
        """
        B = x.shape[0]
        t_batch = t.expand(B)
        u = self.model(x, t_batch)

        return u

    @torch.no_grad()
    def forward(self, batch_size, L, device, return_traj=False):
        """
        x_1: [B, 1, L]

        Integrate from t=1 to t=0.
        """
        x_1 = torch.randn(batch_size, 1, L, device=device)

        t_span = torch.tensor([1.0, 0.0], device=device)

        options = None
        if self.method in ["euler", "midpoint", "rk4"]:
            n_steps = self.n_total_steps
            step_size = 1.0 / n_steps
            options = {"step_size": step_size}

        x_traj = odeint(
            self.ode_func,
            x_1,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options,
        )

        if return_traj:
            return x_traj

        return x_traj[-1]
    
    @torch.no_grad()
    def recover(self, x_t, t, n_steps, return_traj=False):
        device = x_t.device

        t_span = torch.tensor([t, 0.0], device=device)

        options = None
        if self.method in ["euler", "midpoint", "rk4"]:
            step_size = t / n_steps
            options = {"step_size": step_size}

        x_traj = odeint(
            self.ode_func,
            x_t,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options,
        )

        if return_traj:
            return x_traj

        return x_traj[-1]


class FMSamplerChecker(nn.Module):
    """
    Utility sampler for monitoring training

    Given phi_0 data, make phi_{t_f} by FMDiffusion,
    then integrate back to t=0 using RGSampler.

    If t_f is sufficiently small, phi_0 will be successfully recovered.
    """

    def __init__(self, white_noise_flow_1d, sampler):
        super().__init__()

        self.white_noise_flow_1d = white_noise_flow_1d
        self.sampler = sampler

    @torch.no_grad()
    def forward(self, phi_0, t_check=0.5, n_steps=100, return_noisy=False):
        """
        phi_0: [B,1,L]

        return:
            x0_rec: [B,1,L]
        """
        B = phi_0.shape[0]
        device = phi_0.device
        t = torch.full((B,), float(t_check), device=device)
        x_t, _, _ = self.white_noise_flow_1d(phi_0, t)

        x_rec = self.sampler.recover(x_t, float(t_check), n_steps)

        if return_noisy:
            return x_rec, x_t

        return x_rec
