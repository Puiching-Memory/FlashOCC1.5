import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

try:
    import pkg_resources
    import DCNv3 as _dcnv3_cuda

    _dcnv3_version = float(pkg_resources.get_distribution('DCNv3').version)
    has_cuda_kernel = True
except Exception:
    _dcnv3_cuda = None
    _dcnv3_version = 0.0
    has_cuda_kernel = False


class DCNv3Function(Function):

    @staticmethod
    def forward(ctx, input, offset, mask,
                kernel_h, kernel_w, stride_h, stride_w,
                pad_h, pad_w, dilation_h, dilation_w,
                group, group_channels, offset_scale, im2col_step,
                remove_center):
        if not has_cuda_kernel:
            raise RuntimeError('DCNv3 CUDA kernel is not available.')

        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center

        args = [
            input, offset, mask, kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w, group, group_channels,
            offset_scale, im2col_step
        ]
        if remove_center or _dcnv3_version > 1.0:
            args.append(remove_center)

        output = _dcnv3_cuda.dcnv3_forward(*args)
        ctx.save_for_backward(input, offset, mask)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask = ctx.saved_tensors

        args = [
            input, offset, mask, ctx.kernel_h, ctx.kernel_w, ctx.stride_h,
            ctx.stride_w, ctx.pad_h, ctx.pad_w, ctx.dilation_h,
            ctx.dilation_w, ctx.group, ctx.group_channels, ctx.offset_scale,
            grad_output.contiguous(), ctx.im2col_step
        ]
        if ctx.remove_center or _dcnv3_version > 1.0:
            args.append(ctx.remove_center)

        grad_input, grad_offset, grad_mask = _dcnv3_cuda.dcnv3_backward(*args)
        return (
            grad_input, grad_offset, grad_mask,
            None, None, None, None, None, None, None, None,
            None, None, None, None, None, None
        )


def _meshgrid(*args):
    try:
        return torch.meshgrid(*args, indexing='ij')
    except TypeError:
        return torch.meshgrid(*args)


def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w,
                          dilation_h, dilation_w, pad_h=0, pad_w=0,
                          stride_h=1, stride_w=1):
    _, h_in, w_in, _ = spatial_shapes
    h_out = (h_in - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    w_out = (w_in - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = _meshgrid(
        torch.linspace(
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (h_out - 1) * stride_h,
            h_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (w_out - 1) * stride_w,
            w_out,
            dtype=torch.float32,
            device=device))
    ref_y = ref_y.reshape(-1)[None] / h_in
    ref_x = ref_x.reshape(-1)[None] / w_in

    return torch.stack((ref_x, ref_y), -1).reshape(1, h_out, w_out, 1, 2)


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h,
                             dilation_w, group, device):
    _, h_in, w_in, _ = spatial_shapes
    x, y = _meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w,
            kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h,
            kernel_h,
            dtype=torch.float32,
            device=device))
    grid = torch.stack((x / w_in, y / h_in), -1).reshape(-1, 1, 2)
    grid = grid.repeat(1, group, 1).permute(1, 0, 2)
    return grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)


def remove_center_sampling_locations(sampling_locations, kernel_w, kernel_h):
    keep = list(range(sampling_locations.shape[-2]))
    center = (kernel_w * kernel_h - 1) // 2
    keep = [i for i in keep if i != center and (i - center) % (center * 2 + 1) != 0]
    return sampling_locations[:, :, :, keep, :]


def dcnv3_core_pytorch(input, offset, mask, kernel_h, kernel_w, stride_h,
                       stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                       group_channels, offset_scale, remove_center):
    if remove_center and (kernel_h % 2 == 0 or kernel_w % 2 == 0 or kernel_h != kernel_w):
        raise ValueError('remove_center is only compatible with square odd kernel size.')

    input = F.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    n, h_in, w_in, _ = input.shape
    _, h_out, w_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w,
        pad_h, pad_w, stride_h, stride_w)
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    spatial_norm = torch.tensor([w_in, h_in], dtype=offset.dtype, device=input.device)
    spatial_norm = spatial_norm.reshape(1, 1, 1, 2)
    spatial_norm = spatial_norm.repeat(1, 1, 1, group * (kernel_h * kernel_w - remove_center))

    sampling_locations = (ref + grid * offset_scale).repeat(n, 1, 1, 1, 1)
    if remove_center:
        sampling_locations = remove_center_sampling_locations(
            sampling_locations, kernel_w=kernel_w, kernel_h=kernel_h)
    sampling_locations = sampling_locations.flatten(3, 4)
    sampling_locations = sampling_locations + offset * offset_scale / spatial_norm

    num_points = kernel_h * kernel_w - remove_center
    sampling_grids = 2 * sampling_locations - 1

    input_ = input.view(n, h_in * w_in, group * group_channels).transpose(1, 2)
    input_ = input_.reshape(n * group, group_channels, h_in, w_in)

    sampling_grid_ = sampling_grids.view(n, h_out * w_out, group, num_points, 2)
    sampling_grid_ = sampling_grid_.transpose(1, 2).flatten(0, 1)

    sampling_input_ = F.grid_sample(
        input_,
        sampling_grid_,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)

    mask = mask.view(n, h_out * w_out, group, num_points).transpose(1, 2)
    mask = mask.reshape(n * group, 1, h_out * w_out, num_points)
    output = (sampling_input_ * mask).sum(-1).view(n, group * group_channels, h_out * w_out)
    return output.transpose(1, 2).reshape(n, h_out, w_out, -1).contiguous()


class ToChannelsFirst(nn.Module):

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class ToChannelsLast(nn.Module):

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim, norm_layer, in_format='channels_last',
                     out_format='channels_last', eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(ToChannelsFirst())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(ToChannelsLast())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(ToChannelsLast())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(ToChannelsFirst())
    else:
        raise NotImplementedError(f'Unsupported norm layer: {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    if act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    if act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'Unsupported act layer: {act_layer}')


def _is_power_of_2(value):
    if (not isinstance(value, int)) or value < 0:
        raise ValueError(f'invalid input for _is_power_of_2: {value} (type: {type(value)})')
    return (value & (value - 1) == 0) and value != 0


class CenterFeatureScaleModule(nn.Module):

    def forward(self, query, center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        return F.linear(
            query,
            weight=center_feature_scale_proj_weight,
            bias=center_feature_scale_proj_bias).sigmoid()


class _DCNv3Base(nn.Module):

    def __init__(self, channels=64, kernel_size=3, dw_kernel_size=None, stride=1,
                 pad=1, dilation=1, group=4, offset_scale=1.0,
                 act_layer='GELU', norm_layer='LN',
                 center_feature_scale=False, remove_center=False):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')

        group_channels = channels // group
        if not _is_power_of_2(group_channels):
            warnings.warn(
                'You had better set channels in DCNv3 so each group width is a power of 2.',
                stacklevel=2)
        if remove_center and kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size

        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.group = group
        self.group_channels = group_channels
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(channels, norm_layer, 'channels_first', 'channels_last'),
            build_act_layer(act_layer),
        )
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - self.remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - self.remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.zeros(group, dtype=torch.float))
            self.center_feature_scale_module = CenterFeatureScaleModule()

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)
        constant_(self.mask.weight.data, 0.0)
        constant_(self.mask.bias.data, 0.0)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def _mix_center_feature(self, x, x_proj, x1):
        if not self.center_feature_scale:
            return x
        center_feature_scale = self.center_feature_scale_module(
            x1,
            self.center_feature_scale_proj_weight,
            self.center_feature_scale_proj_bias)
        center_feature_scale = center_feature_scale[..., None].repeat(
            1, 1, 1, 1, self.channels // self.group).flatten(-2)
        return x * (1 - center_feature_scale) + x_proj * center_feature_scale


class DCNv3_pytorch(_DCNv3Base):

    def forward(self, input):
        n, h, w, _ = input.shape
        x = self.input_proj(input)
        x_proj = x

        x1 = self.dw_conv(input.permute(0, 3, 1, 2))
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(n, h, w, self.group, -1)
        mask = F.softmax(mask, -1).reshape(n, h, w, -1)

        x = dcnv3_core_pytorch(
            x,
            offset,
            mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale,
            self.remove_center)
        x = self._mix_center_feature(x, x_proj, x1)
        return self.output_proj(x)


class DCNv3(_DCNv3Base):

    def forward(self, input):
        if not has_cuda_kernel:
            return DCNv3_pytorch.forward(self, input)

        n, h, w, _ = input.shape
        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = self.dw_conv(input.permute(0, 3, 1, 2))
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(n, h, w, self.group, -1)
        mask = F.softmax(mask, -1).reshape(n, h, w, -1).type(dtype)

        x = DCNv3Function.apply(
            x,
            offset,
            mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        x = self._mix_center_feature(x, x_proj, x1)
        return self.output_proj(x)
