import triton
import triton.language as tl
import torch


@triton.jit
def _bev_pool_v2_forward_kernel(
    depth_ptr,
    feat_ptr,
    ranks_depth_ptr,
    ranks_feat_ptr,
    ranks_bev_ptr,
    interval_starts_ptr,
    interval_lengths_ptr,
    out_ptr,
    C,
    n_intervals,
    BLOCK_C: tl.constexpr,
):
    pid_interval = tl.program_id(0)
    pid_c = tl.program_id(1)

    if pid_interval >= n_intervals:
        return

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    interval_start = tl.load(interval_starts_ptr + pid_interval)
    interval_length = tl.load(interval_lengths_ptr + pid_interval)
    bev_idx = tl.load(ranks_bev_ptr + interval_start)

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for rel in tl.range(0, interval_length):
        point_idx = interval_start + rel
        depth_idx = tl.load(ranks_depth_ptr + point_idx)
        feat_idx = tl.load(ranks_feat_ptr + point_idx)

        depth_val = tl.load(depth_ptr + depth_idx).to(tl.float32)
        feat_ptrs = feat_ptr + feat_idx * C + offs_c
        feat_vals = tl.load(feat_ptrs, mask=mask_c, other=0.0).to(tl.float32)
        acc += feat_vals * depth_val

    out_ptrs = out_ptr + bev_idx * C + offs_c
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_c)


def kernel_function(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
    total_bev: int,
) -> torch.Tensor:
    if not depth.is_cuda or not feat.is_cuda:
        raise ValueError("depth and feat must be CUDA tensors")
    if depth.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported depth dtype: {depth.dtype}")
    if feat.dtype != depth.dtype:
        raise ValueError("depth and feat must have the same dtype")
    if feat.ndim != 5:
        raise ValueError(f"feat must be [B, N, H, W, C], got {tuple(feat.shape)}")
    if depth.ndim != 5:
        raise ValueError(f"depth must be [B, N, D, H, W], got {tuple(depth.shape)}")

    depth = depth.contiguous()
    feat = feat.contiguous()
    ranks_depth = ranks_depth.contiguous().int()
    ranks_feat = ranks_feat.contiguous().int()
    ranks_bev = ranks_bev.contiguous().int()
    interval_starts = interval_starts.contiguous().int()
    interval_lengths = interval_lengths.contiguous().int()

    channels = feat.shape[-1]
    n_intervals = int(interval_starts.numel())
    out = torch.zeros((int(total_bev), channels), device=feat.device, dtype=torch.float32)

    if n_intervals == 0:
        return out

    block_c = 64
    grid = (n_intervals, triton.cdiv(channels, block_c))
    _bev_pool_v2_forward_kernel[grid](
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
        out,
        channels,
        n_intervals,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )
    return out
