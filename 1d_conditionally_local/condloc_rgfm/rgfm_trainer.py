import torch
import torch.nn as nn
import torch.nn.functional as F


class RGTrainer(nn.Module):
    """
    Real-space local-patch RG flow matching trainer.

    model:
        input : [B, 1, L]  (we take n_channel=1 for now)
        output: [B, 1, L]
    """

    def __init__(self, model, rg_diffusion_real, t_interval):
        super().__init__()

        self.model = model
        self.rg_diffusion_real = rg_diffusion_real
        self.t_s = t_interval[0]
        self.t_f = t_interval[1]

        assert 0.0 <= self.t_s < self.t_f <= 1.0

    def forward(self, phi_0, resizeL):
        """
        phi_0: [B,1,L]

        return:
            loss
        """
        B = phi_0.shape[0]
        device = phi_0.device

        # time interval for the model: t in [t_s, t_f]
        s = torch.rand(B, device=device)
        t = self.t_s + (self.t_f - self.t_s) * s

        # make RG path and target vector field
        # phi_t, u_t: [B, 1, resizeL]
        phi_t, u_t = self.rg_diffusion_real(phi_0, t, resizeL)

        # predict u_t with local time for the model: s=(t-t_s)/(t_f - t_s)
        pred = self.model(phi_t, s)
        loss = F.mse_loss(pred, u_t)

        # check
        assert phi_t.shape == u_t.shape
        assert pred.shape == u_t.shape

        return loss
