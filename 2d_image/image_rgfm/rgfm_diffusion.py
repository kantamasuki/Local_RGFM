"""Renormalization-group flow path for image experiments."""

import numpy as np
import torch
import torch.nn as nn
import torch_dct as dct


class ExponentialRegulator:
    """Exponential regulator kappa(x) = exp(-x)."""

    @staticmethod
    def kappa_inv(y):
        return -np.log(y)

    @staticmethod
    def sqrt_kappa(x):
        return torch.exp(-0.5 * x)

    @staticmethod
    def sqrt_one_minus_kappa(x):
        return torch.sqrt((-torch.expm1(-x)).clamp_min(1e-12))

    @staticmethod
    def velocity_coefficient(x):
        denominator = torch.sqrt((-torch.expm1(-x)).clamp_min(1e-12))
        return x * torch.exp(-0.5 * x) / denominator


class RGDiffusionReal(nn.Module):
    """Construct the RG flow and its velocity field in real space."""

    def __init__(
        self,
        L=64,
        sigma_min=0.1,
        mu_min=0.01,
        m0=1.0,
        mL=0.01,
        reg_name="exp",
    ):
        super().__init__()
        if reg_name != "exp":
            raise ValueError(f"Unknown regulator: {reg_name}")

        self.L = L
        self.regulator = ExponentialRegulator()

        lambda_0, tau = self._make_rg_scale_params(
            L=L,
            sigma_min=sigma_min,
            mu_min=mu_min,
            mL=mL,
        )
        self.register_buffer("Lam0", torch.tensor(lambda_0, dtype=torch.float32))
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.float32))

        k_sq = self._make_massive_k_sq(L, mL)
        self.register_buffer("k_sq", k_sq)
        self.register_buffer("sk", m0 / torch.sqrt(k_sq))

    def _make_rg_scale_params(self, L, sigma_min, mu_min, mL):
        kmax = np.pi * (L - 1) / L * np.sqrt(2.0)
        kmax = np.sqrt(kmax**2 + mL**2)
        kmin = mL

        lambda_0 = kmax / np.sqrt(
            self.regulator.kappa_inv(1.0 - sigma_min**2)
        )
        tau = 2.0 / np.log(
            (kmax**2 / kmin**2)
            * self.regulator.kappa_inv(mu_min**2)
            / self.regulator.kappa_inv(1.0 - sigma_min**2)
        )
        return lambda_0, tau

    @staticmethod
    def _make_massive_k_sq(L, mL):
        modes = torch.arange(L, dtype=torch.float32)
        mode_x, mode_y = torch.meshgrid(modes, modes, indexing="xy")
        k_sq = (mode_x**2 + mode_y**2) * (np.pi / L) ** 2 + mL**2
        return k_sq.view(1, 1, L, L)

    def coefficients(self, t):
        t = t.view(t.shape[0], 1, 1, 1)
        lambda_t_sq = self.Lam0**2 * torch.exp(-2.0 * t / self.tau)
        x = self.k_sq / lambda_t_sq

        sqrt_kappa = self.regulator.sqrt_kappa(x)
        sqrt_one_minus_kappa = self.regulator.sqrt_one_minus_kappa(x)
        velocity_coefficient = self.regulator.velocity_coefficient(x) / self.tau
        return sqrt_kappa, sqrt_one_minus_kappa, velocity_coefficient

    @torch.no_grad()
    def image_size_lifter(self, x_t, t_scalar, small_L, large_L):
        batch_size, channels, height, width = x_t.shape
        if height != small_L or width != small_L:
            raise ValueError(
                f"Expected an image of size {small_L} x {small_L}, "
                f"got {height} x {width}"
            )
        if not small_L < large_L <= self.L:
            raise ValueError(
                f"Expected small_L < large_L <= {self.L}, "
                f"got {small_L} and {large_L}"
            )

        t = torch.full(
            (batch_size,),
            float(t_scalar),
            device=x_t.device,
            dtype=x_t.dtype,
        )
        _, sqrt_one_minus_kappa, _ = self.coefficients(t)

        noise = torch.randn(
            batch_size,
            channels,
            large_L,
            large_L,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        noise_k = dct.dct_2d(noise, norm="ortho")
        x_t_k = dct.dct_2d(x_t, norm="ortho")

        lifted_k = (
            sqrt_one_minus_kappa[:, :, :large_L, :large_L]
            * self.sk[:, :, :large_L, :large_L]
            * noise_k
        )
        lifted_k[:, :, :small_L, :small_L] = x_t_k
        return dct.idct_2d(lifted_k, norm="ortho")

    @torch.no_grad()
    def forward(self, phi_0, t, resizeL=None):
        if resizeL is None:
            resizeL = phi_0.shape[-1]

        sqrt_kappa, sqrt_one_minus_kappa, velocity_coefficient = (
            self.coefficients(t)
        )
        phi_0_k = dct.dct_2d(phi_0, norm="ortho")
        noise_k = dct.dct_2d(torch.randn_like(phi_0), norm="ortho")

        phi_t_k = (
            sqrt_kappa * phi_0_k
            + sqrt_one_minus_kappa * self.sk * noise_k
        )
        u_t_k = velocity_coefficient * (
            -sqrt_one_minus_kappa * phi_0_k
            + sqrt_kappa * self.sk * noise_k
        )

        phi_t = dct.idct_2d(
            phi_t_k[:, :, :resizeL, :resizeL], norm="ortho"
        )
        u_t = dct.idct_2d(
            u_t_k[:, :, :resizeL, :resizeL], norm="ortho"
        )
        return phi_t, u_t

    @torch.no_grad()
    def sample_prior(self, batch_size, resizeL, channels=3, device=None):
        if device is None:
            device = self.k_sq.device

        t = torch.ones(batch_size, device=device)
        _, sqrt_one_minus_kappa, _ = self.coefficients(t)
        noise = torch.randn(
            batch_size,
            channels,
            self.L,
            self.L,
            device=device,
        )
        noise_k = dct.dct_2d(noise, norm="ortho")
        phi_1_k = (
            sqrt_one_minus_kappa[:, :, :resizeL, :resizeL]
            * self.sk[:, :, :resizeL, :resizeL]
            * noise_k[:, :, :resizeL, :resizeL]
        )
        return dct.idct_2d(phi_1_k, norm="ortho")
