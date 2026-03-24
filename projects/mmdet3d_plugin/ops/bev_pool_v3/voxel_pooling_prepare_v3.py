"""voxel_pooling_prepare v3 — fused CUDA kernel implementation.

Migrated from FlashOCC2. Replaces ~10 separate PyTorch ops with:
  1. One CUDA kernel: coordinate transform + bounds check + rank compute
  2. torch.sort (CUB DeviceRadixSort internally)
  3. One CUDA kernel: interval detection
  4. Simultaneously computes feat-sorted intervals for zero-argsort backward

Also provides a pure-PyTorch fallback (voxel_pooling_prepare_v3_pytorch).
"""

from __future__ import annotations

import torch

try:
    from . import bev_pool_v3_ext
except ImportError:
    bev_pool_v3_ext = None

__all__ = ["voxel_pooling_prepare_v3", "voxel_pooling_prepare_v3_pytorch"]


# =====================================================================
#           Helper: compute intervals from sorted keys
# =====================================================================

def _compute_intervals_from_sorted(sorted_keys):
    """Given a sorted int tensor, compute interval starts and lengths."""
    n = sorted_keys.shape[0]
    if n == 0:
        return None, None

    if bev_pool_v3_ext is not None:
        flags = torch.empty(n, dtype=torch.int32, device=sorted_keys.device)
        bev_pool_v3_ext.compute_intervals_v3(sorted_keys.int().contiguous(), flags)
        starts = torch.where(flags == 1)[0].int()
    else:
        kept = torch.ones(n, device=sorted_keys.device, dtype=torch.bool)
        kept[1:] = sorted_keys[1:] != sorted_keys[:-1]
        starts = torch.where(kept)[0].int()

    if starts.numel() == 0:
        return None, None
    lengths = torch.zeros_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = n - starts[-1]
    return starts.contiguous(), lengths.contiguous()


def _compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev):
    """Sort by ranks_feat and compute feat-grouping intervals for backward."""
    order = ranks_feat.long().argsort()
    rf = ranks_feat[order]
    rd = ranks_depth[order]
    rb = ranks_bev[order]

    kept = torch.ones(rf.shape[0], device=rf.device, dtype=torch.bool)
    kept[1:] = rf[1:] != rf[:-1]
    starts = torch.where(kept)[0].int()
    lengths = torch.zeros_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = rf.shape[0] - starts[-1]
    return (rd.int().contiguous(), rf.int().contiguous(), rb.int().contiguous(),
            starts.contiguous(), lengths.contiguous())


# =====================================================================
#                   Fused CUDA prepare (primary path)
# =====================================================================

def voxel_pooling_prepare_v3(coor, grid_lower, grid_size, grid_step, B, N, D, H, W):
    """Fused prepare using CUDA kernel when available, else PyTorch fallback.

    Args:
        coor (Tensor float32): (B*N*D*H*W, 3) world coordinates
        grid_lower (tuple): (lower_x, lower_y, lower_z)
        grid_size  (tuple): (Dx, Dy, Dz)  integer voxel counts
        grid_step  (tuple): (dx, dy, dz)  voxel sizes
        B, N, D, H, W: batch / camera / depth / height / width

    Returns:
        (ranks_bev, ranks_depth, ranks_feat,
         interval_starts, interval_lengths,
         feat_intervals)   — all int32 contiguous tensors, or None on empty
    """
    device = coor.device
    lower_x, lower_y, lower_z = grid_lower
    Dx, Dy, Dz = grid_size
    dx, dy, dz = grid_step
    num_points = B * N * D * H * W

    if bev_pool_v3_ext is not None:
        out_bev   = torch.empty(num_points, dtype=torch.int32, device=device)
        out_depth = torch.empty(num_points, dtype=torch.int32, device=device)
        out_feat  = torch.empty(num_points, dtype=torch.int32, device=device)
        counter   = torch.zeros(1, dtype=torch.int32, device=device)

        n_valid = bev_pool_v3_ext.voxel_pooling_prepare_v3_fused(
            coor.float().contiguous(),
            out_bev, out_depth, out_feat, counter,
            float(lower_x), float(lower_y), float(lower_z),
            float(dx), float(dy), float(dz),
            int(Dx), int(Dy), int(Dz),
            int(B), int(N), int(D), int(H), int(W),
        )

        if n_valid == 0:
            return None, None, None, None, None, None

        ranks_bev   = out_bev[:n_valid]
        ranks_depth = out_depth[:n_valid]
        ranks_feat  = out_feat[:n_valid]
    else:
        # Fallback: pure Python (slow, for CPU debug only)
        return voxel_pooling_prepare_v3_pytorch(
            coor.view(B, N, D, H, W, 3), grid_lower, grid_size, grid_step)

    # Sort by bev rank
    order = ranks_bev.long().argsort()
    ranks_bev   = ranks_bev[order]
    ranks_depth = ranks_depth[order]
    ranks_feat  = ranks_feat[order]

    interval_starts, interval_lengths = _compute_intervals_from_sorted(ranks_bev)
    if interval_starts is None:
        return None, None, None, None, None, None

    feat_intervals = _compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev)

    return (ranks_bev.int().contiguous(),
            ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(),
            interval_starts.int().contiguous(),
            interval_lengths.int().contiguous(),
            feat_intervals)


# =====================================================================
#                   Pure-PyTorch fallback
# =====================================================================

def voxel_pooling_prepare_v3_pytorch(coor, grid_lower, grid_size, grid_step):
    """Pure PyTorch prepare — identical logic to v2, plus feat_intervals.

    Args:
        coor (Tensor): (B, N, D, fH, fW, 3) world coordinates
        grid_lower (tuple): (lower_x, lower_y, lower_z)
        grid_size  (tuple): (Dx, Dy, Dz)
        grid_step  (tuple): (dx, dy, dz)

    Returns: same 6-tuple as voxel_pooling_prepare_v3
    """
    B, N, D, H, W, _ = coor.shape
    device = coor.device
    lower_x, lower_y, lower_z = grid_lower
    Dx, Dy, Dz = grid_size
    dx, dy, dz = grid_step

    # Flatten points
    coor = coor.view(-1, 3)          # (B*N*D*H*W, 3)
    num_points = coor.shape[0]

    # Build depth / feat ranks
    ranks_depth = torch.arange(0, num_points, device=device, dtype=torch.long)
    ranks_feat  = torch.arange(0, B * N * H * W, device=device, dtype=torch.long)
    ranks_feat  = ranks_feat.reshape(B, N, 1, H, W)
    ranks_feat  = ranks_feat.expand(B, N, D, H, W).reshape(-1)

    # Quantize to voxel indices
    coor_x = ((coor[:, 0] - lower_x) / dx).long()
    coor_y = ((coor[:, 1] - lower_y) / dy).long()
    coor_z = ((coor[:, 2] - lower_z) / dz).long()

    kept = ((coor_x >= 0) & (coor_x < Dx) &
            (coor_y >= 0) & (coor_y < Dy) &
            (coor_z >= 0) & (coor_z < Dz))
    if kept.sum() == 0:
        return None, None, None, None, None, None

    coor_x     = coor_x[kept]
    coor_y     = coor_y[kept]
    coor_z     = coor_z[kept]
    ranks_depth = ranks_depth[kept]
    ranks_feat  = ranks_feat[kept]

    ranks_bev = (coor_z * Dy * Dx + coor_y * Dx + coor_x)
    order = ranks_bev.argsort()
    ranks_bev   = ranks_bev[order]
    ranks_depth = ranks_depth[order]
    ranks_feat  = ranks_feat[order]

    kept2 = torch.ones(ranks_bev.shape[0], device=device, dtype=torch.bool)
    kept2[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept2)[0].int()
    if interval_starts.numel() == 0:
        return None, None, None, None, None, None
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

    feat_intervals = _compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev)

    return (ranks_bev.int().contiguous(),
            ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(),
            interval_starts.contiguous(),
            interval_lengths.contiguous(),
            feat_intervals)
