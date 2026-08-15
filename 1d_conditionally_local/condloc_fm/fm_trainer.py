import torch.nn as nn
import torch.nn.functional as F


class FMTrainer(nn.Module):
    """
    Real-space local-patch flow matching trainer.

    model:
        input : [B, 1, L]  (we take n_channel=1 for now)
        output: [B, 1, L]
    """

    def __init__(self, model, white_noise_flow_1d):
        super().__init__()

        self.model = model
        self.flow_1d = white_noise_flow_1d

    def forward(self, phi_0):
        """
        phi_0: [B,1,L]

        return:
            loss
        """
        # make RG path and target vector field
        # phi_t, u_t: [B, 1, L], t: [B,]
        phi_t, u_t, t = self.flow_1d(phi_0)
        pred = self.model(phi_t, t)
        loss = F.mse_loss(pred, u_t)

        return loss
