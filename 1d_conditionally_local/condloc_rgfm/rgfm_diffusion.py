
import torch
import torch.nn as nn
import torch_dct as dct
import numpy as np


class RGParamExp:
    """
    exp-type regulator:
        kappa(x) = exp(-x)
        sqrt(kappa) = exp(-x/2)
        sqrt(1-kappa) = sqrt(1-exp(-x))
    """
    
    def kappa_inv(self, y):
        # y = kappa(x) = exp(-x)
        return -np.log(y)

    def sq_kappa(self, x):
        return torch.exp(-0.5 * x)

    def sq_1m_kappa(self, x):
        # numerically stable sqrt(1 - exp(-x))
        return torch.sqrt((-torch.expm1(-x)).clamp_min(1e-12))

    def c(self, x):
        """
        c(x) = - x kappa'(x) / sqrt(kappa(x)) / sqrt(1-kappa(x))
             = x exp(-x/2) / sqrt(1-exp(-x))
        """
        denom = torch.sqrt((-torch.expm1(-x)).clamp_min(1e-12))
        return x * torch.exp(-0.5 * x) / denom


def make_rg_param(reg_name):
    if reg_name == "exp":
        return RGParamExp()
    else:
        raise ValueError(f"Unknown regulator: {reg_name}")


class RGDiffusion1D(nn.Module):
    """
    Base class for RG flow matching path.

    phi_t(k) = sqrt(kappa_tk) phi_0(k)
             + sqrt(1-kappa_tk) s_k eps(k)

    u_t(k) = d phi_t(k) / dt
           = c_tk [ -sqrt(1-kappa_tk) phi_0(k)
                    + sqrt(kappa_tk) s_k eps(k) ]

    where c_tk = c((k^2+mL^2) / Lambda_t^2) / tau.
    """
    
    def __init__(
        self,
        L=1024,
        sigma_min=0.1,
        mu_min=0.01,
        mL=0.02,
        reg_name="exp"
    ):
        super().__init__()
    
        self.L = L
        self.sigma_min = sigma_min
        self.mu_min = mu_min
        self.mL = mL
        self.m0 = np.sqrt((np.pi/L)**2 + mL**2)
        
        self.rg_param = make_rg_param(reg_name)
        
        Lam0, tau = self._make_rg_scale_params(L, sigma_min, mu_min, mL)
        self.register_buffer("Lam0", torch.tensor(Lam0, dtype=torch.float32))
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.float32))

        k_sq = self._make_massive_k_sq(L, mL)
        self.register_buffer("k_sq", k_sq)

        # noise scale s_k
        sk = self.m0 / torch.sqrt(k_sq)
        self.register_buffer("sk", sk)


    def _make_rg_scale_params(self, L, sigma_min, mu_min, mL):
        kmax = np.sqrt((np.pi*(L-1)/L)**2 + mL**2)
        kmin = mL

        Lam0 = kmax / np.sqrt(
            self.rg_param.kappa_inv(1.0 - sigma_min**2)
        )

        tau = 2.0 / np.log(
            (kmax**2 / kmin**2)
            * self.rg_param.kappa_inv(mu_min**2)
            / self.rg_param.kappa_inv(1.0 - sigma_min**2)
        )

        return Lam0, tau

    def _make_massive_k_sq(self, L, mL):
        k = torch.arange(L, dtype=torch.float32) * np.pi / L
        k_sq = k**2 + mL**2

        return k_sq.view(1, 1, L)

    def lambda_t_sq(self, t):
        """
        t: [B]
        return: [B,1,1]
        """
        t = t.view(t.shape[0], 1, 1)
        return self.Lam0**2 * torch.exp(-2.0 * t / self.tau)

    def coefficients(self, t):
        """
        return:
            sq_kappa:    [B,1,L]
            sq_1m_kappa: [B,1,L]
            ctk:         [B,1,L]
        """
        Lam_t_sq = self.lambda_t_sq(t)
        x = self.k_sq / Lam_t_sq

        sq_kappa = self.rg_param.sq_kappa(x)
        sq_1m_kappa = self.rg_param.sq_1m_kappa(x)
        ctk = self.rg_param.c(x) / self.tau

        return sq_kappa, sq_1m_kappa, ctk

    @torch.no_grad()
    def chain_size_lifter(self, xt, t_scalar, smallL, largeL):
        B, C, L = xt.shape
        device=xt.device
        assert L == smallL
        assert smallL < largeL <= self.L
        
        t = t_scalar * torch.ones(B, device=device)
        _, sq_1m_kappa, _ = self.coefficients(t)
        
        eps = torch.randn(B, C, largeL, device=device)
        eps_k = dct.dct(eps, norm="ortho")
        xtk = dct.dct(xt, norm="ortho")
        
        # higher-k mode: noise 
        # lower-k mode: xtk
        xtk_large = sq_1m_kappa[:, :, :largeL] * self.sk[:, :, :largeL] * eps_k
        xtk_large[:, :, :smallL] = xtk
        xt_large = dct.idct(xtk_large, norm="ortho")
        
        return xt_large
    

class RGDiffusionReal1D(RGDiffusion1D):
    """
    Return real-space (phi_t, u_t)
    
    If the paraemter resizeL is not None,
    make resized image by 
        phi_t →(DCT,ortho)→ phi_tk 
        →(low-k)→ phi_tk[:,:,:resizeL] 
        →(iDCT,ortho)→ phi_t(resized)
    """
    @torch.no_grad()
    def forward(self, phi_0, t, resizeL=None):
        """
        phi_0: [B,C,L]
        t:     [B]

        return:
            phi_t: [B,C,L]
            u_t:   [B,C,L]
        """
        
        if resizeL == None:
            resizeL = phi_0.shape[-1]
        
        sq_kappa, sq_1m_kappa, ctk = self.coefficients(t)

        phi_0_k = dct.dct(phi_0, norm="ortho")

        eps = torch.randn_like(phi_0)
        eps_k = dct.dct(eps, norm="ortho")

        phi_t_k = sq_kappa * phi_0_k + sq_1m_kappa * self.sk * eps_k
        u_t_k = ctk * (-sq_1m_kappa * phi_0_k + sq_kappa * self.sk * eps_k)

        phi_t_k_resize = phi_t_k[:, :, :resizeL]
        u_t_k_resize = u_t_k[:, :, :resizeL]

        phi_t_resize = dct.idct(phi_t_k_resize, norm="ortho")
        u_t_resize = dct.idct(u_t_k_resize, norm="ortho")

        return phi_t_resize, u_t_resize

    @torch.no_grad()
    def sample_prior(self, batch_size, resizeL, channels=1, device=None):
        """
        Sample phi_{t=1}.
        return: [B,C,resizeL]
        """
    
        if device is None:
            device = self.k_sq.device

        t = torch.ones(batch_size, device=device)
        _, sq_1m_kappa, _ = self.coefficients(t)

        eps = torch.randn(batch_size, channels, self.L, device=device)
        eps_k = dct.dct(eps, norm="ortho")

        phi_1_k = sq_1m_kappa[:,:,:resizeL] * self.sk[:,:,:resizeL] * eps_k[:,:,:resizeL]
        phi_1 = dct.idct(phi_1_k, norm="ortho")

        return phi_1
