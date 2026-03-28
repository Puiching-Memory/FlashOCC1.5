from __future__ import annotations

import torch
import torch.nn as nn


B = 1
N = 6
D = 88
H = 16
W = 44
C = 64
DZ = 1
DY = 200
DX = 200
TOTAL_BEV = B * DZ * DY * DX
SEED = 20260326


def build_case(
    batch_size: int = B,
    cams: int = N,
    depth_bins: int = D,
    feat_h: int = H,
    feat_w: int = W,
    channels: int = C,
    bev_z: int = DZ,
    bev_y: int = DY,
    bev_x: int = DX,
    seed: int = SEED,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    total_depth = batch_size * cams * depth_bins * feat_h * feat_w
    total_bev = batch_size * bev_z * bev_y * bev_x

    depth = torch.randn(
        batch_size, cams, depth_bins, feat_h, feat_w, generator=generator, dtype=dtype
    ).to(device)
    feat = torch.randn(
        batch_size, cams, feat_h, feat_w, channels, generator=generator, dtype=dtype
    ).to(device)

    ranks_depth = torch.arange(total_depth, dtype=torch.int32)
    hw = feat_h * feat_w
    dhw = depth_bins * hw
    ranks_feat = ((ranks_depth % hw) + (ranks_depth // dhw) * hw).int()
    ranks_bev = torch.randint(
        0, total_bev, (total_depth,), generator=generator, dtype=torch.int32
    )

    order = torch.argsort(ranks_bev.long())
    ranks_depth = ranks_depth[order].contiguous()
    ranks_feat = ranks_feat[order].contiguous()
    ranks_bev = ranks_bev[order].contiguous()

    kept = torch.ones(ranks_bev.shape[0], dtype=torch.bool)
    kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

    return (
        depth,
        feat,
        ranks_depth.to(device),
        ranks_feat.to(device),
        ranks_bev.to(device),
        interval_starts.to(device),
        interval_lengths.to(device),
        total_bev,
    )


class Model(nn.Module):
    def forward(
        self,
        depth: torch.Tensor,
        feat: torch.Tensor,
        ranks_depth: torch.Tensor,
        ranks_feat: torch.Tensor,
        ranks_bev: torch.Tensor,
        interval_starts: torch.Tensor,
        interval_lengths: torch.Tensor,
        total_bev: int,
    ) -> torch.Tensor:
        del interval_starts, interval_lengths
        channels = feat.shape[-1]
        feat_flat = feat.reshape(-1, channels).float()
        depth_flat = depth.reshape(-1).float()

        gathered_feat = feat_flat[ranks_feat.long()]
        gathered_depth = depth_flat[ranks_depth.long()].unsqueeze(1)
        weighted_feat = gathered_feat * gathered_depth

        out = torch.zeros(
            (int(total_bev), channels), device=feat.device, dtype=torch.float32
        )
        out.index_add_(0, ranks_bev.long(), weighted_feat)
        return out


def get_inputs():
    return list(build_case())


def get_init_inputs():
    return []


def get_benchmark_config():
    return {
        "warmup": 10,
        "repeat": 40,
    }
