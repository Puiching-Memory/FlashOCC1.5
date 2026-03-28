"""Benchmark BEVPool v3 kernel latency and equivalent FPS.

This script compares the original v3 kernel against the patched v3 kernel
using a fixed kernel-only workload and reports:
1. forward latency / FPS
2. forward+backward latency / FPS

FPS here means sample throughput under the chosen batch size:
    FPS = batch_size * 1000 / latency_ms
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
V2_EXT_DIR = ROOT / "projects" / "mmdet3d_plugin" / "ops" / "bev_pool_v2"
V3_EXT_DIR = ROOT / "projects" / "mmdet3d_plugin" / "ops" / "bev_pool_v3"
OLD_V3_SRC_DIR = ROOT / "tmp_bench_bev_pool_v3_old"

sys.path.insert(0, str(V2_EXT_DIR))
sys.path.insert(0, str(V3_EXT_DIR))

import bev_pool_v3_ext as bev_pool_v3_ext_new  # noqa: E402


V3_EXT = bev_pool_v3_ext_new


def compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev):
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
    return rd.contiguous(), rf.contiguous(), rb.contiguous(), starts.contiguous(), lengths.contiguous()


class BevPoolV3Cuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
        feat_intervals=None,
    ):
        ranks_bev = ranks_bev.int()
        if depth.dtype != feat.dtype:
            depth = depth.to(feat.dtype)
        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()
        out = feat.new_zeros(bev_feat_shape)
        V3_EXT.bev_pool_v3_forward(
            depth,
            feat,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )
        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        ctx.feat_intervals = feat_intervals
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors
        fi = ctx.feat_intervals
        if fi is not None:
            rd_fs, rf_fs, rb_fs, starts_fs, lengths_fs = fi
        else:
            rd_fs, rf_fs, rb_fs, starts_fs, lengths_fs = compute_feat_intervals(
                ranks_feat, ranks_depth, ranks_bev
            )
        depth_grad = torch.zeros_like(depth).float()
        feat_grad = torch.zeros_like(feat)
        V3_EXT.bev_pool_v3_backward(
            out_grad.contiguous(),
            depth_grad,
            feat_grad,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            ranks_bev.shape[0],
            rd_fs,
            rf_fs,
            rb_fs,
            starts_fs,
            lengths_fs,
        )
        return depth_grad.to(depth.dtype), feat_grad, None, None, None, None, None, None, None


def bev_pool_v3(
    depth,
    feat,
    ranks_depth,
    ranks_feat,
    ranks_bev,
    bev_feat_shape,
    interval_starts,
    interval_lengths,
    feat_intervals=None,
):
    x = BevPoolV3Cuda.apply(
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
        feat_intervals,
    )
    return x.permute(0, 4, 1, 2, 3).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BEVPool v3 kernel FPS")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cams", type=int, default=6)
    parser.add_argument("--depth-bins", type=int, default=88)
    parser.add_argument("--feat-h", type=int, default=16)
    parser.add_argument("--feat-w", type=int, default=44)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--bev-z", type=int, default=1)
    parser.add_argument("--bev-y", type=int, default=200)
    parser.add_argument("--bev-x", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=60)
    return parser.parse_args()


def build_old_ext():
    build_dir = OLD_V3_SRC_DIR / "build_fps"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="bev_pool_v3_ext_old_bench_fps",
        sources=[
            str(OLD_V3_SRC_DIR / "bev_pool_v3.cpp"),
            str(OLD_V3_SRC_DIR / "bev_pool_v3_cuda.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=str(build_dir),
        verbose=False,
    )


def depth_to_feat_idx(ranks_depth, d, h, w):
    hw = h * w
    dhw = d * hw
    return (ranks_depth % hw) + (ranks_depth // dhw) * hw


def build_case(args: argparse.Namespace, device: str):
    b = args.batch_size
    n = args.cams
    d = args.depth_bins
    h = args.feat_h
    w = args.feat_w
    c = args.channels
    dz = args.bev_z
    dy = args.bev_y
    dx = args.bev_x

    total_depth = b * n * d * h * w
    ranks_depth = torch.arange(total_depth, device=device, dtype=torch.int32)
    ranks_feat = depth_to_feat_idx(ranks_depth.long(), d, h, w).int()
    total_bev = b * dz * dy * dx
    ranks_bev = torch.randint(0, total_bev, (total_depth,), device=device, dtype=torch.int32)
    order = torch.argsort(ranks_bev.long())
    ranks_bev = ranks_bev[order].contiguous()
    ranks_depth = ranks_depth[order].contiguous()
    ranks_feat = ranks_feat[order].contiguous()

    kept = torch.ones(ranks_bev.shape[0], device=device, dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
    feat_intervals = compute_feat_intervals(ranks_feat, ranks_depth, ranks_bev)
    bev_shape = (b, dz, dy, dx, c)
    return ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, feat_intervals, bev_shape


def timed_forward(dtype, ext, case, args):
    global V3_EXT
    V3_EXT = ext
    rd, rf, rb, ist, ilen, fi, bev_shape = case
    b, n, d, h, w, c = (
        args.batch_size,
        args.cams,
        args.depth_bins,
        args.feat_h,
        args.feat_w,
        args.channels,
    )
    depth = torch.randn(b, n, d, h, w, device="cuda", dtype=dtype)
    feat = torch.randn(b, n, h, w, c, device="cuda", dtype=dtype)

    for _ in range(args.warmup):
        bev_pool_v3(depth, feat, rd, rf, rb, bev_shape, ist, ilen, fi)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        bev_pool_v3(depth, feat, rd, rf, rb, bev_shape, ist, ilen, fi)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / args.iters


def timed_fwdbwd(dtype, ext, case, args):
    global V3_EXT
    V3_EXT = ext
    rd, rf, rb, ist, ilen, fi, bev_shape = case
    b, n, d, h, w, c = (
        args.batch_size,
        args.cams,
        args.depth_bins,
        args.feat_h,
        args.feat_w,
        args.channels,
    )
    grad_out = torch.randn(
        b, c, args.bev_z, args.bev_y, args.bev_x, device="cuda", dtype=torch.float32
    )

    def run_once():
        depth = torch.randn(b, n, d, h, w, device="cuda", dtype=dtype, requires_grad=True)
        feat = torch.randn(b, n, h, w, c, device="cuda", dtype=dtype, requires_grad=True)
        out = bev_pool_v3(depth, feat, rd, rf, rb, bev_shape, ist, ilen, fi)
        (out.float() * grad_out).sum().backward()

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run_once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / args.iters


def fps_from_ms(batch_size: int, latency_ms: float) -> float:
    return batch_size * 1000.0 / latency_ms


def main():
    args = parse_args()
    torch.manual_seed(1234)
    assert torch.cuda.is_available(), "CUDA is required"
    device = "cuda"

    case = build_case(args, device)
    ext_old = build_old_ext()
    ext_new = bev_pool_v3_ext_new

    results = []
    for label, ext in (("Original v3", ext_old), ("Patched v3", ext_new)):
        for dtype_name, dtype in (("FP32", torch.float32), ("FP16", torch.float16)):
            fwd_ms = timed_forward(dtype, ext, case, args)
            fwdbwd_ms = timed_fwdbwd(dtype, ext, case, args)
            results.append(
                {
                    "kernel": label,
                    "dtype": dtype_name,
                    "batch_size": args.batch_size,
                    "forward_ms": fwd_ms,
                    "forward_fps": fps_from_ms(args.batch_size, fwd_ms),
                    "fwdbwd_ms": fwdbwd_ms,
                    "fwdbwd_fps": fps_from_ms(args.batch_size, fwdbwd_ms),
                }
            )

    for row in results:
        print("RESULT", json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
