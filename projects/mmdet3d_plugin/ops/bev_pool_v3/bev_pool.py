"""BEV Pool v3 — CUDA backend, supports FP16/BF16.

Migrated from FlashOCC2. Core optimizations vs v2:
  1. Block-per-interval shared-memory tiling -> reduced global memory bandwidth
  2. Unified interval kernel reused for forward + feat backward
  3. Independent depth-backward kernel -> fully parallel over all points
  4. FP16/BF16 input with FP32 accumulation -> 2x bandwidth
  5. Pre-sorted feat intervals -> zero argsort overhead in backward
"""

from __future__ import annotations

import torch

from . import bev_pool_v3_ext

__all__ = ["bev_pool_v3", "TRTBEVPoolv3"]


def _compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev):
    """Re-sort by ranks_feat and compute feat-grouping intervals for backward."""
    order = ranks_feat.argsort()
    rf = ranks_feat[order]
    rd = ranks_depth[order]
    rb = ranks_bev[order]

    kept = torch.ones(rf.shape[0], device=rf.device, dtype=torch.bool)
    kept[1:] = rf[1:] != rf[:-1]
    starts = torch.where(kept)[0].int()
    lengths = torch.zeros_like(starts)
    lengths[:-1] = starts[1:] - starts[:-1]
    lengths[-1] = rf.shape[0] - starts[-1]
    return rd.contiguous(), rf.contiguous(), rb.contiguous(), \
        starts.contiguous(), lengths.contiguous()


class _BevPoolV3Cuda(torch.autograd.Function):
    """Autograd wrapper for the v3 CUDA forward/backward kernels."""

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths,
                feat_intervals=None):
        ranks_bev = ranks_bev.int()
        if depth.dtype != feat.dtype:
            depth = depth.to(feat.dtype)
        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)  # (B, Dz, Dy, Dx, C)

        bev_pool_v3_ext.bev_pool_v3_forward(
            depth, feat, out,
            ranks_depth, ranks_feat, ranks_bev,
            interval_lengths, interval_starts,
        )

        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        ctx.feat_intervals = feat_intervals
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors
        out_grad = out_grad.contiguous()
        n_points = ranks_bev.shape[0]

        # Use pre-computed feat intervals if available (zero argsort overhead)
        fi = ctx.feat_intervals
        if fi is not None:
            rd_fs, rf_fs, rb_fs, starts_fs, lengths_fs = fi
        else:
            rd_fs, rf_fs, rb_fs, starts_fs, lengths_fs = \
                _compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev)

        depth_grad = torch.zeros_like(depth).float()
        feat_grad = torch.zeros_like(feat)

        # Signature: bev_pool_v3_backward(
        #   out_grad, depth_grad, feat_grad,
        #   depth, feat,
        #   ranks_depth, ranks_feat, ranks_bev, n_points,
        #   ranks_depth_fs, ranks_feat_fs, ranks_bev_fs,
        #   interval_starts_fs, interval_lengths_fs)
        bev_pool_v3_ext.bev_pool_v3_backward(
            out_grad, depth_grad, feat_grad,
            depth, feat,
            ranks_depth, ranks_feat, ranks_bev, n_points,
            rd_fs, rf_fs, rb_fs,
            starts_fs, lengths_fs,
        )

        return depth_grad.to(depth.dtype), feat_grad, \
            None, None, None, None, None, None, None


def bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev,
               bev_feat_shape, interval_starts, interval_lengths,
               feat_intervals=None):
    """BEV pool v3 forward.

    Args:
        depth (Tensor): (B, N, D, fH, fW)
        feat (Tensor): (B, N, fH, fW, C)
        ranks_depth (Tensor int32): (N_points,)
        ranks_feat (Tensor int32): (N_points,)
        ranks_bev (Tensor int32): (N_points,)
        bev_feat_shape (tuple): (B, Dz, Dy, Dx, C)
        interval_starts (Tensor int32): (N_intervals,)
        interval_lengths (Tensor int32): (N_intervals,)
        feat_intervals: optional pre-computed feat intervals for backward

    Returns:
        Tensor: (B, C, Dz, Dy, Dx)
    """
    x = _BevPoolV3Cuda.apply(
        depth, feat, ranks_depth, ranks_feat, ranks_bev,
        bev_feat_shape, interval_starts, interval_lengths,
        feat_intervals)
    return x.permute(0, 4, 1, 2, 3).contiguous()


class TRTBEVPoolv3(torch.autograd.Function):
    """TensorRT-compatible BEV pooling v3."""

    @staticmethod
    def symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                 interval_starts, interval_lengths,
                 output_height=128, output_width=128, output_z=1):
        return g.op("mmdeploy::bev_pool_v3",
                    depth, feat, ranks_depth, ranks_feat, ranks_bev,
                    interval_starts, interval_lengths,
                    output_height_i=output_height,
                    output_width_i=output_width,
                    output_z_i=output_z)

    @staticmethod
    def forward(g, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                interval_starts, interval_lengths,
                output_height=128, output_width=128, output_z=1):
        feat = feat.unsqueeze(0)
        depth = depth.unsqueeze(0)
        shape = (depth.shape[0], output_z, output_height, output_width,
                 feat.shape[-1])
        bev_feat = bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               shape, interval_starts, interval_lengths)
        if output_z == 1:
            bev_feat = bev_feat.squeeze(2).permute(0, 2, 3, 1)
        return bev_feat
