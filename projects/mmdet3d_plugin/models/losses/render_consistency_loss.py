import torch
import torch.nn.functional as F


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return values.mean()


def volume_rendering_consistency_loss(
    occ_pred: torch.Tensor = None,
    occ_logits: torch.Tensor = None,
    voxel_semantics: torch.Tensor = None,
    mask_camera: torch.Tensor = None,
    empty_idx: int = 17,
    max_rays: int = 4096,
    eps: float = 1e-6,
    free_weight: float = 1.0,
    depth_weight: float = 1.0,
    hit_weight: float = 1.0,
):
    """NeRF-like occupancy consistency losses on sampled rays.

    Args:
        occ_pred: (B, Dx, Dy, Dz, C) occupancy logits.
        voxel_semantics: (B, Dx, Dy, Dz) semantic labels.
        mask_camera: (B, Dx, Dy, Dz) visible mask.
        empty_idx: class index of empty voxels.
        max_rays: max sampled (x, y) columns per batch.
        eps: numerical stability epsilon.

    Returns:
        dict with scalar losses: {'free', 'depth', 'hit'}.
    """
    if occ_pred is None:
        occ_pred = occ_logits
    if occ_pred is None or voxel_semantics is None or mask_camera is None:
        raise ValueError('occ_pred/voxel_semantics/mask_camera must be provided.')
    B, Dx, Dy, Dz, _ = occ_pred.shape
    device = occ_pred.device

    probs = occ_pred.softmax(dim=-1)
    p_empty = probs[..., empty_idx].clamp(min=eps, max=1.0 - eps)
    p_occ = (1.0 - p_empty).clamp(min=eps, max=1.0 - eps)

    z_ids = torch.arange(Dz, dtype=torch.float32, device=device)

    rays_per_sample = max(1, max_rays // max(B, 1))

    free_losses = []
    depth_losses = []
    hit_losses = []

    for b in range(B):
        sem = voxel_semantics[b].long()                 # (Dx, Dy, Dz)
        vis = mask_camera[b].bool()                     # (Dx, Dy, Dz)
        valid = vis & (sem != 255)

        valid_cols = valid.any(dim=-1)                 # (Dx, Dy)
        col_idx = valid_cols.nonzero(as_tuple=False)   # (R_all, 2)
        if col_idx.numel() == 0:
            continue

        if col_idx.shape[0] > rays_per_sample:
            perm = torch.randperm(col_idx.shape[0], device=device)[:rays_per_sample]
            col_idx = col_idx[perm]

        ix = col_idx[:, 0]
        iy = col_idx[:, 1]

        sem_ray = sem[ix, iy, :]                        # (R, Dz)
        valid_ray = valid[ix, iy, :]                    # (R, Dz)
        p_empty_ray = p_empty[b, ix, iy, :]             # (R, Dz)
        p_occ_ray = p_occ[b, ix, iy, :]                 # (R, Dz)

        gt_occ = (sem_ray != empty_idx) & valid_ray
        has_hit = gt_occ.any(dim=-1)                    # (R,)
        first_hit = gt_occ.float().argmax(dim=-1)       # (R,)

        alpha = p_occ_ray * valid_ray.float()
        trans = torch.cumprod(
            torch.cat([torch.ones(alpha.shape[0], 1, device=device), 1.0 - alpha + eps], dim=-1),
            dim=-1,
        )[:, :Dz]
        ray_weights = trans * alpha
        expected_depth = (ray_weights * z_ids.unsqueeze(0)).sum(dim=-1)

        if has_hit.any():
            target_depth = first_hit[has_hit].float()
            depth_l1 = (expected_depth[has_hit] - target_depth).abs() / max(Dz - 1, 1)
            depth_losses.append(_safe_mean(depth_l1))

            hit_prob = p_occ_ray[has_hit, :].gather(1, first_hit[has_hit].unsqueeze(-1)).squeeze(-1)
            hit_losses.append(_safe_mean(F.binary_cross_entropy(hit_prob, torch.ones_like(hit_prob), reduction='none')))

            before_hit = (z_ids.unsqueeze(0) < first_hit.unsqueeze(-1)) & valid_ray & has_hit.unsqueeze(-1)
            if before_hit.any():
                free_prob = p_empty_ray[before_hit]
                free_losses.append(_safe_mean(F.binary_cross_entropy(free_prob, torch.ones_like(free_prob), reduction='none')))

        no_hit = (~has_hit).unsqueeze(-1) & valid_ray
        if no_hit.any():
            free_prob_no_hit = p_empty_ray[no_hit]
            free_losses.append(_safe_mean(F.binary_cross_entropy(free_prob_no_hit, torch.ones_like(free_prob_no_hit), reduction='none')))

    def _stack_or_zero(loss_list):
        if len(loss_list) == 0:
            return occ_pred.new_tensor(0.0)
        return torch.stack(loss_list).mean()

    return {
        'free': _stack_or_zero(free_losses) * free_weight,
        'depth': _stack_or_zero(depth_losses) * depth_weight,
        'hit': _stack_or_zero(hit_losses) * hit_weight,
    }


def render_consistency_loss(*args, **kwargs):
    """Backward-compatible alias."""
    return volume_rendering_consistency_loss(*args, **kwargs)
