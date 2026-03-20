"""ELAN (Efficient Layer Aggregation Network) backbone from YOLOv7.

Replaces the standard ResNet backbone with a pure-Python ELAN architecture
that uses multi-path feature aggregation for richer gradient flow.

Reference: YOLOv7 — Trainable bag-of-freebies sets new state-of-the-art for
real-time object detectors (https://arxiv.org/abs/2207.02696)

Stage output channels (width_mult=1.0):
    P2(/4)=256, P3(/8)=512, P4(/16)=1024, P5(/32)=1024
"""
import torch
import torch.nn as nn
from mmdet3d.models import BACKBONES


def _autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def _fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d for inference."""
    fusedconv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, groups=conv.groups, bias=True,
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Conv2d + BN + SiLU."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ElanBlock(nn.Module):
    """ELAN 多路径特征聚合模块.

    两条 1×1 分支从输入分叉, 其中一条经过 ``num_convs`` 个 3×3 卷积.
    取 [branch_a, branch_b, conv2_out, conv4_out] 拼接后经 1×1 过渡卷积.

    Args:
        in_ch:  输入通道数
        mid_ch: 分支内部通道数
        out_ch: 输出通道数 (= 4 * mid_ch 时拼接后等宽, 再经 1×1 映射)
        num_convs: 串联 3×3 卷积数, 每隔一个取特征用于拼接. 默认 4.
    """

    def __init__(self, in_ch, mid_ch, out_ch, num_convs=4):
        super().__init__()
        self.branch_a = Conv(in_ch, mid_ch, 1, 1)
        self.branch_b = Conv(in_ch, mid_ch, 1, 1)
        self.convs = nn.ModuleList([Conv(mid_ch, mid_ch, 3, 1) for _ in range(num_convs)])
        # concat: branch_a + branch_b + conv2_out + conv4_out = 4 * mid_ch
        self.transition = Conv(mid_ch * 4, out_ch, 1, 1)

    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)
        outs = [b]
        for conv in self.convs:
            outs.append(conv(outs[-1]))
        # 取 branch_a, branch_b(=outs[0]), conv2(=outs[2]), conv4(=outs[4])
        cat = torch.cat([a, outs[0], outs[2], outs[4]], dim=1)
        return self.transition(cat)


class MPTransition(nn.Module):
    """MaxPool 双路下采样过渡模块.

    Path A: MaxPool(2) → Conv 1×1
    Path B: Conv 1×1  → Conv 3×3 stride 2
    输出 = Concat([path_b, path_a]) → 2 * mid_ch 通道
    """

    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_a = Conv(in_ch, mid_ch, 1, 1)
        self.conv_b1 = Conv(in_ch, mid_ch, 1, 1)
        self.conv_b2 = Conv(mid_ch, mid_ch, 3, 2)

    def forward(self, x):
        a = self.conv_a(self.mp(x))
        b = self.conv_b2(self.conv_b1(x))
        return torch.cat([b, a], dim=1)


# ---------------------------------------------------------------------------
#  Full backbone
# ---------------------------------------------------------------------------

@BACKBONES.register_module()
class Elan(nn.Module):
    """YOLOv7-ELAN backbone (pure Python, 无 YAML 依赖).

    网络拓扑:
        Stem  (/4)  → ElanBlock P2(256) → MPTransition → ElanBlock P3(512)
              → MPTransition → ElanBlock P4(1024) → MPTransition → ElanBlock P5(1024)

    Args:
        out_indices (tuple[int]): 要返回的 stage 索引, 0=P2 1=P3 2=P4 3=P5.
            默认 ``(2, 3)`` 返回 P4(/16, 1024ch) 和 P5(/32, 1024ch).
        in_channels (int): 输入图像通道数. 默认 3.
        pretrained (str | None): 预训练权重路径.
    """

    # Stage 输出通道: P2=256, P3=512, P4=1024, P5=1024
    STAGE_CHANNELS = (256, 512, 1024, 1024)

    def __init__(self, out_indices=(2, 3), in_channels=3, pretrained=None):
        super().__init__()
        self.out_indices = out_indices

        # ---- Stem: 3 → 128, /4 ----
        self.stem = nn.Sequential(
            Conv(in_channels, 32, 3, 1),
            Conv(32, 64, 3, 2),   # /2
            Conv(64, 64, 3, 1),
            Conv(64, 128, 3, 2),  # /4
        )

        # ---- Stage P2 (/4): 128 → 256 ----
        self.stage0 = ElanBlock(128, 64, 256)

        # ---- Transition P2→P3 + Stage P3 (/8): 256→256→512 ----
        self.trans1 = MPTransition(256, 128)   # out = 256
        self.stage1 = ElanBlock(256, 128, 512)

        # ---- Transition P3→P4 + Stage P4 (/16): 512→512→1024 ----
        self.trans2 = MPTransition(512, 256)   # out = 512
        self.stage2 = ElanBlock(512, 256, 1024)

        # ---- Transition P4→P5 + Stage P5 (/32): 1024→1024→1024 ----
        self.trans3 = MPTransition(1024, 512)  # out = 1024
        self.stage3 = ElanBlock(1024, 256, 1024)

        if pretrained is not None:
            self._load_pretrain(pretrained)

    def _load_pretrain(self, pretrained):
        ckpt = torch.load(pretrained, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        own = self.state_dict()
        matched = {k: v for k, v in ckpt.items()
                   if k in own and v.shape == own[k].shape}
        self.load_state_dict(matched, strict=False)
        print(f'[Elan] Loaded {len(matched)}/{len(own)} params from {pretrained}')

    def forward(self, x):
        x = self.stem(x)

        feats = []
        x = self.stage0(x);  feats.append(x)   # P2 /4   256
        x = self.stage1(self.trans1(x));  feats.append(x)  # P3 /8   512
        x = self.stage2(self.trans2(x));  feats.append(x)  # P4 /16  1024
        x = self.stage3(self.trans3(x));  feats.append(x)  # P5 /32  1024

        return [feats[i] for i in self.out_indices]

    def fuse(self):
        """Fuse Conv+BN for faster inference."""
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = _fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
