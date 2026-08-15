# patch_maker.py
# Optimized PatchMaker for the common case D == stride and L % D == 0.

import torch


class PatchMaker:
    def __init__(self, L=64, P=24, D=8, stride=8, normalize_coord=True):
        self.L = L
        self.P = P
        self.D = D
        self.stride = stride
        self.normalize_coord = normalize_coord

        assert P <= L
        assert D <= P
        assert (L - D) % stride == 0

        self.n_side = (L - D) // stride + 1
        self.n_patch = self.n_side ** 2

        self.patch_infos = self._make_patch_infos()
        self.coord_templates = self._make_coord_templates()       # [N, 2, P, P], CPU
        self.mask_templates = self._make_mask_templates()         # [N, 1, P, P], CPU

        # Indices used by vectorized/gather-based implementations.
        self.condition_linear_indices = self._make_condition_linear_indices()  # [N*P*P]
        self.target_linear_indices = self._make_target_linear_indices()        # [N, D*D]
        self.target_in_patch_indices = self._make_target_in_patch_indices()    # [N, D*D]

        # Small lazy cache so repeated ODE evaluations do not repeatedly call .to(device).
        self._tensor_cache = {}

    @staticmethod
    def _clip(x, lo, hi):
        return max(lo, min(x, hi))

    def _make_patch_infos(self):
        patch_infos = []
        for iy in range(self.n_side):
            for ix in range(self.n_side):
                tx0 = ix * self.stride
                ty0 = iy * self.stride
                tx1 = tx0 + self.D
                ty1 = ty0 + self.D

                cx0 = tx0 + self.D // 2 - self.P // 2
                cy0 = ty0 + self.D // 2 - self.P // 2
                cx0 = self._clip(cx0, 0, self.L - self.P)
                cy0 = self._clip(cy0, 0, self.L - self.P)
                cx1 = cx0 + self.P
                cy1 = cy0 + self.P

                rx0 = tx0 - cx0
                ry0 = ty0 - cy0
                rx1 = rx0 + self.D
                ry1 = ry0 + self.D

                patch_infos.append({
                    "target":    (tx0, tx1, ty0, ty1),
                    "condition": (cx0, cx1, cy0, cy1),
                    "relative":  (rx0, rx1, ry0, ry1),
                })
        return patch_infos

    def _make_coord_templates(self):
        coords = []
        for info in self.patch_infos:
            cx0, cx1, cy0, cy1 = info["condition"]
            xs = torch.arange(cx0, cx1)
            ys = torch.arange(cy0, cy1)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            xx = xx.float()
            yy = yy.float()
            if self.normalize_coord:
                xx = 2.0 * xx / (self.L - 1) - 1.0
                yy = 2.0 * yy / (self.L - 1) - 1.0
            coords.append(torch.stack([xx, yy], dim=0))
        return torch.stack(coords, dim=0)

    def _make_mask_templates(self):
        masks = []
        for info in self.patch_infos:
            rx0, rx1, ry0, ry1 = info["relative"]
            mask = torch.zeros(1, self.P, self.P)
            mask[:, ry0:ry1, rx0:rx1] = 1.0
            masks.append(mask)
        return torch.stack(masks, dim=0)

    def _make_condition_linear_indices(self):
        inds = []
        for info in self.patch_infos:
            cx0, cx1, cy0, cy1 = info["condition"]
            xs = torch.arange(cx0, cx1)
            ys = torch.arange(cy0, cy1)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            inds.append((yy * self.L + xx).reshape(-1))
        return torch.cat(inds, dim=0).long()

    def _make_target_linear_indices(self):
        """Indices into flattened full image y[..., L*L], shape [N, D*D]."""
        inds = []
        for info in self.patch_infos:
            tx0, tx1, ty0, ty1 = info["target"]
            xs = torch.arange(tx0, tx1)
            ys = torch.arange(ty0, ty1)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            inds.append((yy * self.L + xx).reshape(-1))
        return torch.stack(inds, dim=0).long()

    def _make_target_in_patch_indices(self):
        """Indices into flattened P*P patch, shape [N, D*D]."""
        inds = []
        for info in self.patch_infos:
            rx0, rx1, ry0, ry1 = info["relative"]
            xs = torch.arange(rx0, rx1)
            ys = torch.arange(ry0, ry1)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            inds.append((yy * self.P + xx).reshape(-1))
        return torch.stack(inds, dim=0).long()

    def __len__(self):
        return self.n_patch

    def _cached(self, name, device, dtype=None):
        """Move small static tensors to device/dtype once and reuse them."""
        key = (name, str(device), str(dtype))
        if key in self._tensor_cache:
            return self._tensor_cache[key]

        tensor = getattr(self, name)
        if dtype is None:
            moved = tensor.to(device=device)
        else:
            moved = tensor.to(device=device, dtype=dtype)
        self._tensor_cache[key] = moved
        return moved

    def clear_cache(self):
        self._tensor_cache.clear()

    def make_input_batch(self, x):
        """
        x: [B, 3, L, L]
        return: [B*N, 5, P, P]
        """
        B, C, H, W = x.shape
        assert C == 3
        assert H == self.L and W == self.L

        x_flat = x.reshape(B, C, self.L * self.L)
        idx = self._cached("condition_linear_indices", x.device, None)

        cond = torch.gather(
            x_flat,
            dim=2,
            index=idx.view(1, 1, -1).expand(B, C, -1),
        )  # [B, C, N*P*P]

        cond = cond.view(B, C, self.n_patch, self.P, self.P)
        cond = cond.permute(0, 2, 1, 3, 4)  # [B, N, 3, P, P]

        coords = self._cached("coord_templates", x.device, x.dtype)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1, -1)

        inp = torch.cat([cond, coords], dim=2)
        return inp.reshape(B * self.n_patch, 5, self.P, self.P)

    def make_target_batch(self, y):
        """
        Vectorized target construction.

        y: [B, C, L, L]
        return: [B*N, C, P, P]
        target is zero outside the target region.
        """
        B, C, H, W = y.shape
        assert H == self.L and W == self.L

        # Gather true D x D target patches from y.
        y_flat = y.reshape(B, C, self.L * self.L)
        target_idx = self._cached("target_linear_indices", y.device, None)  # [N, D*D]
        target_vals = torch.gather(
            y_flat,
            dim=2,
            index=target_idx.reshape(1, 1, -1).expand(B, C, -1),
        )  # [B, C, N*D*D]
        target_vals = target_vals.view(B, C, self.n_patch, self.D * self.D)
        target_vals = target_vals.permute(0, 2, 1, 3)  # [B, N, C, D*D]

        # Scatter each D x D target into the corresponding P x P patch location.
        out = torch.zeros(
            B, self.n_patch, C, self.P * self.P,
            device=y.device, dtype=y.dtype,
        )
        patch_idx = self._cached("target_in_patch_indices", y.device, None)  # [N, D*D]
        scatter_idx = patch_idx.view(1, self.n_patch, 1, self.D * self.D).expand(B, -1, C, -1)
        out.scatter_(dim=3, index=scatter_idx, src=target_vals)

        out = out.view(B, self.n_patch, C, self.P, self.P)
        return out.reshape(B * self.n_patch, C, self.P, self.P)

    def make_mask_batch(self, batch_size, device=None, dtype=torch.float32):
        if device is None:
            device = self.mask_templates.device
        masks = self._cached("mask_templates", device, dtype)
        masks = masks.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        return masks.reshape(batch_size * self.n_patch, 1, self.P, self.P)

    def make_training_batch(self, x, y):
        inp = self.make_input_batch(x)
        target = self.make_target_batch(y)
        mask = self.make_mask_batch(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        return inp, target, mask

    def reconstruct_from_pred_batch(self, pred_batch, batch_size):
        # Keep the general slow version for overlapping patches.
        BN, C, P1, P2 = pred_batch.shape
        assert P1 == self.P and P2 == self.P
        assert BN == batch_size * self.n_patch

        pred_batch = pred_batch.reshape(batch_size, self.n_patch, C, self.P, self.P)
        full = torch.zeros(batch_size, C, self.L, self.L, device=pred_batch.device, dtype=pred_batch.dtype)
        weight = torch.zeros(batch_size, 1, self.L, self.L, device=pred_batch.device, dtype=pred_batch.dtype)

        for i, info in enumerate(self.patch_infos):
            tx0, tx1, ty0, ty1 = info["target"]
            rx0, rx1, ry0, ry1 = info["relative"]
            pred_target = pred_batch[:, i, :, ry0:ry1, rx0:rx1]
            full[:, :, ty0:ty1, tx0:tx1] += pred_target
            weight[:, :, ty0:ty1, tx0:tx1] += 1.0

        return full / weight.clamp_min(1.0)

    def reconstruct_from_pred_batch_simple(self, pred_batch, batch_size):
        """
        Fully vectorized reconstruction for D == stride and L % D == 0.
        """
        assert self.stride == self.D
        assert self.L % self.D == 0
        assert self.n_side == self.L // self.D

        BN, C, P1, P2 = pred_batch.shape
        assert P1 == self.P and P2 == self.P
        assert BN == batch_size * self.n_patch

        pred = pred_batch.reshape(batch_size, self.n_patch, C, self.P * self.P)
        patch_idx = self._cached("target_in_patch_indices", pred_batch.device, None)  # [N, D*D]

        pred_target = torch.gather(
            pred,
            dim=3,
            index=patch_idx.view(1, self.n_patch, 1, self.D * self.D).expand(batch_size, -1, C, -1),
        )  # [B, N, C, D*D]

        patches = pred_target.view(batch_size, self.n_patch, C, self.D, self.D)
        patches = patches.reshape(batch_size, self.n_side, self.n_side, C, self.D, self.D)
        full = patches.permute(0, 3, 1, 4, 2, 5)
        return full.reshape(batch_size, C, self.L, self.L)
