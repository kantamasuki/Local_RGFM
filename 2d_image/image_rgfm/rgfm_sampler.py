# rgfm_sampler.py

import torch
import torch.nn as nn
from torchdiffeq import odeint


class RGSamplerReal(nn.Module):
    """
    Real-space local-patch sampler with image size resizeL.

    Integrates from t=t_f to t=t_s using torchdiffeq.odeint.

    model:
        input : [B*N, 5, P, P]
        output: [B*N, 3, P, P]
    """

    def __init__(
        self,
        model,
        patch_maker,
        t_interval,  # [t_s, t_f]
        n_total_steps=300,
        method="midpoint",   # RK2
        rtol=1e-5,
        atol=1e-5,
    ):
        super().__init__()

        self.model = model
        self.patch_maker = patch_maker
        self.t_s = t_interval[0]
        self.t_f = t_interval[1]
        self.n_total_steps = n_total_steps
        self.method = method
        self.rtol = rtol
        self.atol = atol

        assert 0.0 <= self.t_s < self.t_f <= 1.0
        assert n_total_steps > 0

    def ode_func(self, s, x):
        """
        s: scalar tensor
        x: [B, 3, L, L]

        return:
            u: [B, 3, L, L]
        """
        B = x.shape[0]

        s_batch = s.expand(B)

        inp = self.patch_maker.make_input_batch(x)  # [B*N, 5, P, P]
        s_patch = s_batch.repeat_interleave(self.patch_maker.n_patch)

        pred_patch = self.model(inp, s_patch)       # [B*N, 3, P, P]

        u = self.patch_maker.reconstruct_from_pred_batch_simple(
            pred_patch,
            batch_size=B,
        )

        return (self.t_f - self.t_s) * u

    @torch.no_grad()
    def forward(self, x_tf, return_traj=False):
        """
        x_tf: [B, 3, L, L]

        Integrate from t_f to t_s.
        """
        device = x_tf.device

        t_span = torch.tensor([1.0, 0.0], device=device)

        options = None
        if self.method in ["euler", "midpoint", "rk4"]:
            n_steps = self.n_total_steps
            step_size = 1.0 / n_steps
            options = {"step_size": step_size}

        x_traj = odeint(
            self.ode_func,
            x_tf,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options,
        )

        if return_traj:
            return x_traj

        return x_traj[-1]


class RGSamplerTotal(nn.Module):
    """
    Total sampler:
        sample images from small-L to large-L models
    """

    def __init__(self, sampler_list, L_list, rg_diffusion_real):
        super().__init__()
        
        assert len(sampler_list) == len(L_list)
        
        for i in range(len(sampler_list)):
            assert sampler_list[i].patch_maker.L == L_list[i]
        for i in range(len(sampler_list)-1):
            assert L_list[i] > L_list[i+1]
            assert abs(sampler_list[i].t_f - sampler_list[i+1].t_s) < 1e-6
        
        self.sampler_list = sampler_list
        self.L_list = L_list
        self.rg_diffusion_real = rg_diffusion_real

    @torch.no_grad()
    def forward(self, batch_size, channels=3, device=None, return_intermediate=False):
        """
        return:
            x0: [B,C,L,L]
        """
        
        intermediate = []
        
        # Sample real-space prior noise at the smallest resolution at t=1.
        Lnow = self.L_list[-1]
        x_tf = self.rg_diffusion_real.sample_prior(batch_size, Lnow)
        assert x_tf.shape == (batch_size, 3, Lnow, Lnow)
        intermediate.append(x_tf)
        
        for i in reversed(range(len(self.L_list))):
            # revert from t_f to t_s with i-th sampler
            sampler = self.sampler_list[i]
            x_ts = sampler(x_tf)
            intermediate.append(x_ts)
            
            if i > 0:
                L_now = self.L_list[i]
                L_large = self.L_list[i-1]
                t_now = sampler.t_s
                x_ts = self.rg_diffusion_real.image_size_lifter(x_ts, t_now, L_now, L_large)
                x_tf = x_ts
                intermediate.append(x_tf)
            
            if i==0:
                x0 = x_ts

        if return_intermediate:
            return x0, intermediate
        else:
            return x0


class RGSamplerChecker(nn.Module):
    """
    Utility sampler for monitoring i-th model during traininng

    Given phi_0 data, make phi_{t_f} by RGDiffusion,
    then integrate back to t=s using RGSampler.

    This checks whether the i-th model can invert the RG path on [t_s, t_f].
    """

    def __init__(self, rg_diffusion_real, sampler):
        super().__init__()

        self.rg_diffusion_real = rg_diffusion_real
        self.sampler = sampler

    @torch.no_grad()
    def forward(self, phi_0, return_noisy=False):
        """
        phi_0: [B,3,L,L]

        return:
            x0_rec: [B,3,L,L]
        """
        B = phi_0.shape[0]
        device = phi_0.device

        t_f = self.sampler.t_f
        t_s = self.sampler.t_s
        Lnow = self.sampler.patch_maker.L
        assert t_s < t_f

        t = torch.full((B,), float(t_f), device=device)
        x_tf, _ = self.rg_diffusion_real(phi_0, t, resizeL=Lnow)

        x_ts = self.sampler(x_tf)

        if return_noisy:
            return x_ts, x_tf

        return x_ts
