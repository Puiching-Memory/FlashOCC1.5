import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from mmcv.runner import BaseModule
from mmdet3d.models import BACKBONES

from DCNv4 import DCNv4


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x.div(keep_prob) * random_tensor


class StemLayer(nn.Module):
    """Stem layer: two 3x3 conv with stride 2 for 4x downsampling."""

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, stride=2, padding=1)
        self.norm1 = nn.LayerNorm(mid)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid, out_channels, 3, stride=2, padding=1)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(self.norm1(x))
        x = x.permute(0, 3, 1, 2)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DownsampleLayer(nn.Module):
    """Downsample by 2x with a 3x3 conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class InternImageLayer(nn.Module):
    """Single DCNv4 layer: LN -> DCNv4 -> residual -> LN -> MLP -> residual."""

    def __init__(self, channels, group, kernel_size=3, mlp_ratio=4.0,
                 drop_path=0.0, offset_scale=1.0, dw_kernel_size=None,
                 center_feature_scale=False):
        super().__init__()
        self.channels = channels
        self.norm1 = nn.LayerNorm(channels)
        self.dcn = DCNv4(
            channels=channels,
            kernel_size=kernel_size,
            group=group,
            pad=kernel_size // 2,
            offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size,
            center_feature_scale=center_feature_scale,
        )
        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        shortcut = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_dcn = self.dcn(self.norm1(shortcut), (H, W))
        x = shortcut + self.drop_path(x_dcn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class InternImageStage(nn.Module):
    """A stage of InternImage layers with optional downsampling."""

    def __init__(self, channels, depth, group, kernel_size=3, mlp_ratio=4.0,
                 drop_path=0.0, downsample=None, offset_scale=1.0,
                 dw_kernel_size=None, center_feature_scale=False, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        dp = drop_path if isinstance(drop_path, (list, tuple)) else [drop_path] * depth
        self.layers = nn.ModuleList([
            InternImageLayer(
                channels=channels, group=group, kernel_size=kernel_size,
                mlp_ratio=mlp_ratio, drop_path=dp[i],
                offset_scale=offset_scale, dw_kernel_size=dw_kernel_size,
                center_feature_scale=center_feature_scale,
            )
            for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x):
        for layer in self.layers:
            if self.with_cp and x.requires_grad:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x_out = x
        x_down = self.downsample(x) if self.downsample is not None else x
        return x_out, x_down


@BACKBONES.register_module()
class FlashInternImage(BaseModule):
    """InternImage backbone built on DCNv4 operators.

    Args:
        in_channels (int): Input image channels. Default: 3.
        channels (int): Base channel count. Default: 64.
        depths (tuple[int]): Number of layers per stage
            (InternImage-T default: (4, 4, 18, 4)).
        groups (tuple[int]): DCNv4 group count per stage. Default: (4, 8, 16, 32).
        kernel_size (int): DCNv4 kernel size. Default: 3.
        mlp_ratio (float): MLP expansion ratio. Default: 4.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        out_indices (tuple[int]): Output stage indices. Default: (2, 3).
        offset_scale (float): DCNv4 offset scale. Default: 1.0.
        dw_kernel_size (int | None): Depth-wise kernel for offset. Default: 5.
        center_feature_scale (bool): Whether to use center feature scaling.
        with_cp (bool): Use gradient checkpointing. Default: False.
        init_cfg: Initialization config for BaseModule.
    """

    def __init__(
        self,
        in_channels=3,
        channels=64,
        depths=(4, 4, 18, 4),
        groups=(4, 8, 16, 32),
        kernel_size=3,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        out_indices=(2, 3),
        offset_scale=1.0,
        dw_kernel_size=5,
        center_feature_scale=False,
        with_cp=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        num_stages = len(depths)

        self.stem = StemLayer(in_channels, channels)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        stage_channels = [channels * (2 ** i) for i in range(num_stages)]

        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_stages):
            downsample = DownsampleLayer(stage_channels[i], stage_channels[i + 1]) \
                if i < num_stages - 1 else None
            stage = InternImageStage(
                channels=stage_channels[i],
                depth=depths[i],
                group=groups[i],
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                offset_scale=offset_scale,
                dw_kernel_size=dw_kernel_size,
                center_feature_scale=center_feature_scale,
                with_cp=with_cp,
            )
            self.stages.append(stage)
            if i in out_indices:
                self.norms.append(nn.LayerNorm(stage_channels[i]))
            else:
                self.norms.append(nn.Identity())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x_out, x = stage(x)
            if i in self.out_indices:
                B, C, H, W = x_out.shape
                out = x_out.permute(0, 2, 3, 1).reshape(B, -1, C)
                out = self.norms[i](out)
                out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs
