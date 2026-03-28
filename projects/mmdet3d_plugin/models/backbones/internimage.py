import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import re

from mmcv.runner import BaseModule
from mmdet3d.models import BACKBONES

from .dcnv3 import DCNv3

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None


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
    """Single InternImage layer with configurable deformable core op."""

    def __init__(self, channels, group, kernel_size=3, mlp_ratio=4.0,
                 drop_path=0.0, offset_scale=1.0, dw_kernel_size=None,
                 center_feature_scale=False, remove_center=False,
                 core_op='DCNv4', post_norm=False, layer_scale=None):
        super().__init__()
        self.channels = channels
        self.core_op = core_op
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        self.norm1 = nn.LayerNorm(channels)

        if core_op == 'DCNv4':
            self.dcn = self._build_dcnv4(
                channels=channels,
                kernel_size=kernel_size,
                group=group,
                offset_scale=offset_scale,
                dw_kernel_size=dw_kernel_size,
                center_feature_scale=center_feature_scale)
        elif core_op == 'DCNv3':
            self.dcn = DCNv3(
                channels=channels,
                kernel_size=kernel_size,
                group=group,
                pad=kernel_size // 2,
                offset_scale=offset_scale,
                dw_kernel_size=dw_kernel_size,
                center_feature_scale=center_feature_scale,
                remove_center=remove_center,
            )
        else:
            raise ValueError(f'Unsupported core_op: {core_op}')

        self.norm2 = nn.LayerNorm(channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.layer_scale:
            self.layer_scale1 = nn.Parameter(layer_scale * torch.ones(channels))
            self.layer_scale2 = nn.Parameter(layer_scale * torch.ones(channels))

    @staticmethod
    def _build_dcnv4(channels, kernel_size, group, offset_scale,
                     dw_kernel_size, center_feature_scale):
        try:
            from DCNv4 import DCNv4
        except Exception as exc:
            raise ImportError(
                'DCNv4 is unavailable. Please install the DCNv4 package in a CUDA-enabled '
                'environment, or switch core_op to DCNv3.'
            ) from exc

        return DCNv4(
            channels=channels,
            kernel_size=kernel_size,
            group=group,
            pad=kernel_size // 2,
            offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size,
            center_feature_scale=center_feature_scale,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        if self.post_norm:
            x_dcn_in = x
        else:
            x_dcn_in = self.norm1(x)

        if self.core_op == 'DCNv4':
            x_dcn = self.dcn(x_dcn_in.reshape(B, H * W, C), (H, W)).reshape(B, H, W, C)
        else:
            x_dcn = self.dcn(x_dcn_in)

        if self.post_norm:
            x_dcn = self.norm1(x_dcn)

        if self.layer_scale:
            x = x + self.drop_path(self.layer_scale1 * x_dcn)
        else:
            x = x + self.drop_path(x_dcn)

        if self.post_norm:
            x_mlp = self.norm2(self.mlp(x))
        else:
            x_mlp = self.mlp(self.norm2(x))

        if self.layer_scale:
            x = x + self.drop_path(self.layer_scale2 * x_mlp)
        else:
            x = x + self.drop_path(x_mlp)
        return x.permute(0, 3, 1, 2).contiguous()


class InternImageStage(nn.Module):
    """A stage of InternImage layers with optional downsampling."""

    def __init__(self, channels, depth, group, kernel_size=3, mlp_ratio=4.0,
                 drop_path=0.0, downsample=None, offset_scale=1.0,
                 dw_kernel_size=None, center_feature_scale=False,
                 remove_center=False, core_op='DCNv4', post_norm=False,
                 layer_scale=None, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        dp = drop_path if isinstance(drop_path, (list, tuple)) else [drop_path] * depth
        self.layers = nn.ModuleList([
            InternImageLayer(
                channels=channels, group=group, kernel_size=kernel_size,
                mlp_ratio=mlp_ratio, drop_path=dp[i],
                offset_scale=offset_scale, dw_kernel_size=dw_kernel_size,
                center_feature_scale=center_feature_scale,
                remove_center=remove_center, core_op=core_op,
                post_norm=post_norm, layer_scale=layer_scale,
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
    """InternImage backbone with configurable DCNv3/DCNv4 core operators.

    Args:
        in_channels (int): Input image channels. Default: 3.
        channels (int): Base channel count. Default: 64.
        depths (tuple[int]): Number of layers per stage
            (InternImage-T default: (4, 4, 18, 4)).
        groups (tuple[int]): Group count per stage. Default: (4, 8, 16, 32).
        kernel_size (int): Deformable kernel size. Default: 3.
        mlp_ratio (float): MLP expansion ratio. Default: 4.0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        out_indices (tuple[int]): Output stage indices. Default: (2, 3).
        offset_scale (float): Offset scale. Default: 1.0.
        dw_kernel_size (int | None): Depth-wise kernel for offset branch. Default: 5.
        center_feature_scale (bool): Whether to use center feature scaling.
        remove_center (bool): Whether to remove center sampling locations for DCNv3.
        core_op (str): Deformable core op, one of ``DCNv4`` or ``DCNv3``.
        post_norm (bool): Whether to use post normalization like official InternImage-B/L.
        layer_scale (float | None): Optional layer scale init value.
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
        remove_center=False,
        core_op='DCNv4',
        post_norm=False,
        layer_scale=None,
        pretrained=None,
        with_cp=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        self.pretrained = pretrained
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
                remove_center=remove_center,
                core_op=core_op,
                post_norm=post_norm,
                layer_scale=layer_scale,
                with_cp=with_cp,
            )
            self.stages.append(stage)
            if i in out_indices:
                self.norms.append(nn.LayerNorm(stage_channels[i]))
            else:
                self.norms.append(nn.Identity())

        self._init_weights()
        if self.pretrained:
            self._load_pretrained(self.pretrained)

    def _init_weights(self):
        for name, m in self.named_modules():
            if '.dcn.' in name or name.startswith('dcn.'):
                continue
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

    @staticmethod
    def _normalize_norm_key(key):
        return re.sub(r'(norm\d*|norm|res_post_norm[12])\.(?:0|1|2)\.', r'\1.', key)

    def _map_official_key(self, key):
        for prefix in ('model.', 'internimage.', 'backbone.'):
            if key.startswith(prefix):
                key = key[len(prefix):]

        key = self._normalize_norm_key(key)

        if key.startswith('patch_embed.'):
            key = key.replace('patch_embed.', 'stem.', 1)
            return key

        if key.startswith('levels.'):
            key = key.replace('.blocks.', '.layers.')
            key = key.replace('.downsample.', '.downsample.')

            if '.layers.' in key:
                key = key.replace('.mlp.fc1.', '.mlp.0.')
                key = key.replace('.mlp.fc2.', '.mlp.2.')
                return key.replace('levels.', 'stages.', 1)

            if '.downsample.' in key:
                return key.replace('levels.', 'stages.', 1)

            if '.norm.' in key:
                return re.sub(r'^levels\.(\d+)\.norm\.', r'norms.\1.', key)

        return None

    def _load_checkpoint_state_dict(self, checkpoint_path):
        if checkpoint_path.endswith('.safetensors'):
            if load_safetensors is None:
                raise ImportError('safetensors is required to load .safetensors checkpoints.')
            return load_safetensors(checkpoint_path, device='cpu')

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model', 'module'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        return checkpoint

    def _load_pretrained(self, checkpoint_path):
        official_state_dict = self._load_checkpoint_state_dict(checkpoint_path)
        model_state_dict = self.state_dict()

        converted_state_dict = {}
        skipped_prefixes = (
            'head.',
            'fc_norm.',
            'clip_projector.',
            'dcnv3_head_x4.',
            'dcnv3_head_x3.',
            'conv_head.',
            'pos_drop.',
            'avgpool.',
        )
        skipped_contains = (
            '.post_norms.',
            '.res_post_norm1.',
            '.res_post_norm2.',
        )

        for key, value in official_state_dict.items():
            if key.startswith(skipped_prefixes) or any(token in key for token in skipped_contains):
                continue

            mapped_key = self._map_official_key(key)
            if mapped_key is None or mapped_key not in model_state_dict:
                continue

            if model_state_dict[mapped_key].shape != value.shape:
                continue
            converted_state_dict[mapped_key] = value

        incompatible = self.load_state_dict(converted_state_dict, strict=False)
        print(
            'FlashInternImage pretrained load:',
            f'checkpoint={checkpoint_path},',
            f'loaded={len(converted_state_dict)},',
            f'missing={len(incompatible.missing_keys)},',
            f'unexpected={len(incompatible.unexpected_keys)}'
        )

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
