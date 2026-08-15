"""Training objective for 2D local RG flow matching."""

import torch
import torch.nn as nn


class RGTrainer(nn.Module):
    """
    Real-space local-patch RG flow matching trainer.

    model:
        input : [B*N, 5, P, P]
                3 image channels + 2 coordinate channels
        output: [B*N, 3, P, P]

    patch_maker:
        must provide make_training_batch(phi_t, u_t)
        returning:
            inp    : [B*N, 5, P, P]
            target : [B*N, 3, P, P]
            mask   : [B*N, 1, P, P]
    """

    def __init__(self, model, rg_diffusion, patch_maker, t_interval):
        super().__init__()

        self.model = model
        self.rg_diffusion = rg_diffusion
        self.patch_maker = patch_maker
        self.t_s = t_interval[0]
        self.t_f = t_interval[1]

        assert 0.0 <= self.t_s < self.t_f <= 1.0

    def forward(self, phi_0, resizeL):
        """
        phi_0: [B,3,L,L]

        return:
            loss
        """
        B = phi_0.shape[0]
        device = phi_0.device

        # train time interval: t in [t_s, t_f]
        t = self.t_s + (self.t_f - self.t_s) * torch.rand(B, device=device)

        # make RG path and target vector field
        # phi_t, u_t: [B, 3, resizeL, resizeL]
        phi_t, u_t = self.rg_diffusion(phi_0, t, resizeL)

        # [B,3,resizeL,resizeL] -> [B*N,5,P,P], [B*N,3,P,P], [B*N,1,P,P]
        # N=(resizeL/D)**2
        inp, target, mask = self.patch_maker.make_training_batch(phi_t, u_t)

        # repeat t for all patches
        s = (t - self.t_s) / (self.t_f - self.t_s)
        s_patch = s.repeat_interleave(self.patch_maker.n_patch)
        pred = self.model(inp, s_patch)

        # mask loss only on target patch region
        loss_map = (pred - target) ** 2
        denom = mask.sum() * pred.shape[1]
        loss = (loss_map * mask).sum() / denom.clamp_min(1.0)

        return loss
