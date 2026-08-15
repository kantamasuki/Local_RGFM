# fm_trainer.py

import torch
import torch.nn as nn


class FMTrainerReal(nn.Module):
    """
    Real-space local-patch flow matching trainer.

    model:
        input : [B*N, 5, P, P]
                3 image channels + 2 coordinate channels
        output: [B*N, 3, P, P]

    patch_maker:
        make_training_batch(x_t, u_t)
        returns:
            inp    : [B*N, 5, P, P]
            target : [B*N, 3, P, P]
            mask   : [B*N, 1, P, P]
    """

    def __init__(self, model, fm_diffusion, patch_maker):
        super().__init__()

        self.model = model
        self.fm_diffusion = fm_diffusion
        self.patch_maker = patch_maker

    def forward(self, x0):
        """
        x0: [B, 3, L, L]
        """
        B = x0.shape[0]
        device = x0.device

        # Standard FM uses t in [0, 1]
        t = torch.rand(B, device=device)

        # full-image path and vector field
        x_t, u_t = self.fm_diffusion(x0, t)

        # local patch training batch
        inp, target, mask = self.patch_maker.make_training_batch(x_t, u_t)

        # repeat t for all patches
        t_patch = t.repeat_interleave(self.patch_maker.n_patch)

        pred = self.model(inp, t_patch)

        # masked mean MSE on target patch
        loss_map = (pred - target) ** 2
        denom = mask.sum() * pred.shape[1]
        loss = (loss_map * mask).sum() / denom.clamp_min(1.0)

        return loss
