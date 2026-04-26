import math
import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import autopad

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
    from einops import rearrange, einsum

except:
    pass


try:
    from torch.cuda.amp import autocast
    import torch.distributions as td
    from torch.utils.checkpoint import checkpoint
    import antialiased_cnns

except:
    pass

try:
    from timm.models.vision_transformer import Attention
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_

except:
    pass

try:
    from typing import Optional, Sequence
    from natten.functional import na2d_av
    from mmengine.model import BaseModule
    from mmcv.cnn import ConvModule, build_norm_layer

except:
    pass

try:
    import torch_dct as dct
    from collections import OrderedDict
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d
except:
    pass


from ultralytics.nn.modules.block import C2f, C3, SPPF, C3k2, C3k, Bottleneck, ADown, ABlock, AAttn
from ultralytics.nn.other_modules.conv import (SCConv, ScConv, AKConv, DropPath, DySnakeConv, GhostModuleV3, PConv,
                                               gnconv, WTConv2d, CARAFE, DySample, Downsizing)
from ultralytics.nn.other_modules.attention import Attention_xy, iRMB, OmniAttention, LSKblock

from ultralytics.nn.modules.conv import (Conv, Conv2, DWConv, DWConvTranspose2d, ConvTranspose, RepConv, LightConv,
                                         GhostConv, )

# __all——Trans__ = (
#     "TransformerEncoderLayer", "TransformerLayer", "TransformerBlock", "MLPBlock", "LayerNorm2d", "AIFI",
#     "DeformableTransformerDecoder", "DeformableTransformerDecoderLayer", "MSDeformAttn", "MLP",
# )

# __all--Conv__ = (
#     "Conv", "Conv2", "LightConv", "DWConv", "DWConvTranspose2d", "ConvTranspose", "Focus", "GhostConv",
#     "ChannelAttention", "SpatialAttention", "CBAM", "Concat", "RepConv",
# )

# __all--Block__ = (
#     "DFL", "HGBlock", "HGStem", "SPP", "SPPF", "C1", "C2", "C3", "C2f", "C2fAttn", "ImagePoolingAttn",
#    "ContrastiveHead", "BNContrastiveHead", "C3x", "C3TR", "C3Ghost", "GhostBottleneck", "Bottleneck", "BottleneckCSP",
#     "Proto", "RepC3", "ResNetLayer", "RepNCSPELAN4", "ELAN1", "ADown", "AConv", "SPPELAN", "CBFuse", "CBLinear",
#     "C3k2", "C2fPSA", "C2PSA", "RepVGGDW", "CIB", "C2fCIB", "Attention", "PSA","SCDown",
# )


# _____________________________________ C3k2_DySnakeConv ___________________________#
class Bottleneck_DySnakeConv(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = DySnakeConv(c_, c2, 3)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) + self.cv3(self.cv1(x)) if self.add else self.cv2(self.cv1(x)) + self.cv3(self.cv1(x))


class C3k_DySnakeConv(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_DySnakeConv(c_, c_) for _ in range(n)))


class C3k2_DySnakeConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_DySnakeConv(self.c, self.c) if c3k else Bottleneck_DySnakeConv(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_DySnakeConv----------------------------------
class C3_DySnakeConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DySnakeConv(c_, c_) for _ in range(n)))


# -------------------------C2f_SMFA----------------------------------
class C2f_DySnakeConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_DySnakeConv(self.c, self.c) for _ in range(n))


# -------------------------SMFA----------------------------------

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)


# _____________________________________ C3k2_SMFA ___________________________#
class C3k_SMFA(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(SMFA(c_) for _ in range(n)))


class C3k2_SMFA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_FRFN(self.c, self.c) if c3k else SMFA(self.c,) for _ in range(n)
        )


# -------------------------C3_SMFA----------------------------------
class C3_SMFA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SMFA(c_) for _ in range(n)))


# -------------------------C2f_SMFA----------------------------------
class C2f_SMFA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(SMFA(self.c) for _ in range(n))


# -------------------------FRFN----------------------------------

class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # bs x hw x c
        c, bs, hh, hw = x.size()
        # hh = int(math.sqrt(hw))
        #
        # # spatial restore
        # x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        x1, x2, = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hw)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) (c) -> b c h w ', h=hh, w=hw)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=hh, w=hw)
        x = x_1 * x_2

        x = self.linear2(x)
        # x = self.eca(x)

        return rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hw)


# _____________________________________ C3k2_FRFN ___________________________#
class C3k_FRFN(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(FRFN(c_, 2*c_) for _ in range(n)))

class C3k2_FRFN(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_FRFN(self.c, 2*self.c) if c3k else FRFN(self.c, 2*self.c) for _ in range(n)
        )


# -------------------------C3_FRFN----------------------------------
class C3_FRFN(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FRFN(c_, 2*c_) for _ in range(n)))


# -------------------------C2f_FRFN----------------------------------
class C2f_FRFN(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(FRFN(self.c, 2*self.c) for _ in range(n))


# _____________________________________ C3k2_InceptionDWConv2d ___________________________#
class C3k_InceptionDWConv2d(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(InceptionDWConv2d(c_) for _ in range(n)))

class C3k2_InceptionDWConv2d(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_InceptionDWConv2d(self.c, self.c) if c3k else InceptionDWConv2d(self.c,) for _ in range(n)
        )


# -------------------------C3_InceptionDWConv2d----------------------------------
class C3_InceptionDWConv2d(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(InceptionDWConv2d(c_,) for _ in range(n)))


# -------------------------C2f_InceptionDWConv2d----------------------------------
class C2f_InceptionDWConv2d(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(InceptionDWConv2d(self.c, ) for _ in range(n))


# -------------------------Inception_CD_Ghost----------------------------------
class Inception_CD_Ghost(nn.Module):
    """ Inception depthweise convolution
    """
    pass
    # def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
    #     super().__init__()
    #
    #     gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
    #     self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
    #     self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
    #                               groups=gc)
    #     self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
    #                               groups=gc)
    #     self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
    #     self.bn = nn.BatchNorm2d(in_channels)
    #
    # def forward(self, x):
    #     x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
    #     return self.bn(torch.cat(
    #         (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
    #         dim=1,
    #     ))


# -------------------------ConvNeXt----------------------------------
class PartialConv2d(nn.Module):
    r"""
    Conduct convolution on partial channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 conv_ratio=1.0,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs,
                 ):
        super().__init__()
        in_chs = int(in_channels * conv_ratio)
        out_chs = int(out_channels * conv_ratio)
        gps = int(groups * conv_ratio) or 1  # groups should be at least 1
        self.conv = nn.Conv2d(in_chs, out_chs,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=gps, bias=bias,
                              **kwargs,
                              )
        self.split_indices = (in_channels - in_chs, in_chs)

    def forward(self, x):
        identity, conv = torch.split(x, self.split_indices, dim=1)
        return torch.cat(
            (identity, self.conv(conv)),
            dim=1,
        )


class Block_ConvNeXt(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, kernel_size=7,
                 drop_path=0., layer_scale_init_value=1e-6,
                 conv_fn=nn.Conv2d,
                 ):
        super().__init__()
        self.dwconv = conv_fn(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, out_chans=32, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 kernel_sizes=7, conv_fns=nn.Conv2d,
                 **kwargs,
                 ):
        super().__init__()
        depths = (3,)
        dims = (out_chans, )
        num_stages = len(depths)
        # num_stages = len(depths)
        self.num_stages = num_stages

        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes] * num_stages
        if not isinstance(conv_fns, (list, tuple)):
            conv_fns = [conv_fns] * num_stages

        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            # nn.LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_stages - 1):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[Block_ConvNeXt(dim=dims[i], drop_path=dp_rates[cur + j],
                        kernel_size=kernel_sizes[i],
                        layer_scale_init_value=layer_scale_init_value,
                        conv_fn=conv_fns[i],
                        ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        return x  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


# -------------------------MetaNeXt----------------------------------
class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        from timm.models.layers import to_2tuple

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_p = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_p(x) + shortcut
        return x


class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        x = self.blocks(x)
        return x


class MetaNeXt(nn.Module):
    """ MetaNeXt
        A PyTorch impl of : `InceptionNeXt: When Inception Meets ConvNeXt`  - https://arxiv.org/pdf/2203.xxxxx.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalziation layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chans=3,
            out_chans=32,
            # depths=(3, 3, 9, 3),
            depths=(3,),
            dims=(96, 192, 384),
            token_mixers=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            **kwargs,
    ):
        super().__init__()
        # dims = [*dims, out_chans]
        dims = (out_chans,)
        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layer(dims[0])
        )

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class InceptionNeXtBlock(MetaNeXt):

    def __init__(
            self,
            in_chans=3,
            out_chans=32,
            # depths=(3, 3, 9, 3),
            depths=(3,),
            dims=(96, 192, 384),
            token_mixers=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            **kwargs,
    ):
        super().__init__()
        # dims = [*dims, out_chans]
        dims = (out_chans,)
        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layer(dims[0])
        )

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


# _____________________________________ C2f_AkConv ___________________________#
class Bottleneck_AkConv(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = AKConv(c_, c2, 1, 3)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) + self.cv3(self.cv1(x)) if self.add else self.cv2(self.cv1(x)) + self.cv3(self.cv1(x))


class C2f_AkConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_AkConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))


# _____________________________________ C2f_ScConv ___________________________#
class Bottleneck_ScConv(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1_2 = ScConv(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) + self.cv3(self.cv1(x)) if self.add else self.cv2(self.cv1(x)) + self.cv3(self.cv1(x))


class C2f_ScConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_ScConv(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))


# _____________________________________ C3k2_ScConv ___________________________#
class C3k_ScConv(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_ScConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_ScConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_ScConv(self.c, self.c, shortcut, g) for _ in range(n)
        )


# _____________________________________ C3k2_AKConv ___________________________#
class C3k_AKConv(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_AkConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_AKConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_ScConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_AkConv(self.c, self.c, shortcut, g) for _ in range(n)
        )


# _____________________________________ C3_AKConv ___________________________#
class C3_AKConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_AkConv(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))


# _____________________________________ C3_ScConv ___________________________#
class C3_ScConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

# _____________________________________RCRep2A____________________________#
class Rep2ABlock(nn.Module):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # mid channels
        self.cv1 = RepConv(c1, c_, 3)
        self.att = Attention_xy(c_)
        self.cv2 = RepConv(c_, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.att(self.cv1(x))) if self.add else self.cv2(self.att(self.cv1(x)))


class RCRep2A(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # self.cv_m = Conv(self.c, self.c, 3)
        self.cv_m = GhostConv(self.c, self.c, 3)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Rep2ABlock(self.c, self.c, shortcut, e=e) for _ in range(n))

    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, 1)
        y = list([self.cv_m(x1), x1, x2])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# _____________________________________ Rep2ABlock_AkConv ___________________________#
class Rep2ABlock_AkConv(Rep2ABlock):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv1 = AKConv(c1, c_, 1, 3)


class RCRep2A_AKConv(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Rep2ABlock_AkConv(self.c, self.c, shortcut, e=e) for _ in range(n))


# _____________________________________ RCRep2A_InceptionSWConv2d ___________________________#
class RCRep2A_InceptionDWConv2d(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(InceptionDWConv2d(self.c) for _ in range(n))


# _____________________________________ RCRep2A_FRFN ___________________________#
class RCRep2A_FRFN(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FRFN(self.c, 2*self.c) for _ in range(n))


# _____________________________________ RCRep2A_SMFA ___________________________#
class RCRep2A_SMFA(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SMFA(self.c) for _ in range(n))


# _____________________________________ RCRep2A_DySnakeConv ___________________________#
class RCRep2A_DySnakeConv(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DySnakeConv(self.c, self.c) for _ in range(n))


# _____________________________________SPPF_WD____________________________#
class SPPF_WD(SPPF):
    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__(c1, c2, k)
        c_ = c1 // 2  # hidden channels
        self.cv1 = LightConv(c1, c_, 1, 1)
        # self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = ConvTranspose(c_ * 4, c2, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=k-2*i, stride=1, padding=(k-2*i) // 2) for i in range(3)])

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m[i](y[-1]) for i in range(3))
        return self.cv2(torch.cat(y, 1))


# _____________________________________HLADP____________________________#
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RepVGGBlocks(nn.Module):
    def __init__(self, channel, fuse_block_num=3) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            *[RepVGGBlock(channel, channel) for _ in range(fuse_block_num)],
        )

    def forward(self, x):
        return self.conv(x)


class align_3In(nn.Module):
    def __init__(self, c1, c2, size_sign):
        super().__init__()
        c_ = c2//2
        self.ss = size_sign
        self.convs = nn.ModuleList(
            [nn.Conv2d(i, c_, kernel_size=3, stride=1, padding=1) for i in c1])
        self.conv2 = Conv(3*c_, c2)

    def forward(self, x):
        y = list([])
        target_size = x[self.ss].shape[2:]

        for i, x1 in enumerate(x):
            if x1.shape[-1] > target_size[-1]:
                x1 = F.adaptive_avg_pool2d(x1, (target_size[0], target_size[1]))
            elif x1.shape[-1] < target_size[-1]:
                x1 = F.interpolate(x1, size=(target_size[0], target_size[1]),
                                  mode='bilinear', align_corners=True)

            y.append(self.convs[i](x1))
        return self.conv2(torch.cat(y, dim=1))


# _____________________________________SCBottleneck____________________________#
# https://gitcode.com/MCG-NKU/SCNet/blob/master/scnet.py
class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, c1, c2, stride=1, downsample=None, cardinality=1, bottleneck_width=32, avd=False, dilation=1, is_first=False, norm_layer=nn.BatchNorm2d):
        super(SCBottleneck, self).__init__()
        group_width = int(c2 * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(c1, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(c1, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            dilation=dilation, groups=cardinality, pooling_r=self.pooling_r)

        self.conv3 = nn.Conv2d(
            group_width * 2, c2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(c2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual if out.shape == residual.shape else out
        out = self.relu(out)

        return out


# _____________________________________ C3k2_SCConv ___________________________#
class Bottleneck_SCConv(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = SCConv(c_, c2, 1)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) + self.cv3(self.cv1(x)) if self.add else self.cv2(self.cv1(x)) + self.cv3(self.cv1(x))


class C3k_SCConv(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_SCConv(c_, c_) for _ in range(n)))


class C3k2_SCConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_SCConv(self.c, self.c) if c3k else Bottleneck_SCConv(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_SCConv----------------------------------
class C3_SCConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_SCConv(c_, c_) for _ in range(n)))


# -------------------------C2f_SCConv----------------------------------
class C2f_SCConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_SCConv(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_SCConv ___________________________#
class Rep2ABlock_SCConv(nn.Module):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # mid channels
        self.cv1 = SCConv(c1, c_, 1)
        self.att = Attention_xy(c_)
        self.cv2 = RepConv(c_, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.att(self.cv1(x))) if self.add else self.cv2(self.att(self.cv1(x)))


class RCRep2A_SCConv(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_SCConv(self.c, self.c) for _ in range(n))


# _____________________________________ C3k2_ScConv ___________________________#
class Bottleneck_ScConv(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = ScConv(c_, )

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_ScConv(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_) for _ in range(n)))


class C3k2_ScConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_ScConv(self.c, self.c) if c3k else Bottleneck_ScConv(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_ScConv----------------------------------
class C3_ScConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_) for _ in range(n)))


# -------------------------C2f_ScConv----------------------------------
class C2f_ScConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_ScConv(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_ScConv ___________________________#
class Rep2ABlock_ScConv(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = ScConv(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_ScConv(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_ScConv(self.c, self.c) for _ in range(n))


# _____________________________________ C3k2_iRMB ___________________________#
class Bottleneck_iRMB(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = iRMB(c_, c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_iRMB(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_iRMB(c_, c_) for _ in range(n)))


class C3k2_iRMB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_iRMB(self.c, self.c) if c3k else Bottleneck_iRMB(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_iRMB----------------------------------
class C3_iRMB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_iRMB(c_, c_) for _ in range(n)))


# -------------------------C2f_iRMB----------------------------------
class C2f_iRMB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_iRMB(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_iRMB ___________________________#
class Rep2ABlock_iRMB(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = iRMB(c_, c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_iRMB(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_iRMB(self.c, self.c) for _ in range(n))


# _____________________________________ PKIBlock ___________________________#

def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int, float): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 2, 3, 1])


class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.permute([0, 3, 1, 2])


# class GSiLU(BaseModule):
#     """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
#     def __init__(self):
#         super().__init__()
#         self.adpool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         return x * torch.sigmoid(self.adpool(x))


# class CAA(BaseModule):
#     """Context Anchor Attention"""
#     def __init__(
#             self,
#             channels: int,
#             h_kernel_size: int = 11,
#             v_kernel_size: int = 11,
#             norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
#             act_cfg: Optional[dict] = dict(type='SiLU'),
#             init_cfg: Optional[dict] = None,
#     ):
#         super().__init__(init_cfg)
#         self.avg_pool = nn.AvgPool2d(7, 1, 3)
#         self.conv1 = ConvModule(channels, channels, 1, 1, 0,
#                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
#         self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
#                                  (0, h_kernel_size // 2), groups=channels,
#                                  norm_cfg=None, act_cfg=None)
#         self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
#                                  (v_kernel_size // 2, 0), groups=channels,
#                                  norm_cfg=None, act_cfg=None)
#         self.conv2 = ConvModule(channels, channels, 1, 1, 0,
#                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
#         return attn_factor
#
#
# class ConvFFN(BaseModule):
#     """Multi-layer perceptron implemented with ConvModule"""
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: Optional[int] = None,
#             hidden_channels_scale: float = 4.0,
#             hidden_kernel_size: int = 3,
#             dropout_rate: float = 0.,
#             add_identity: bool = True,
#             norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
#             act_cfg: Optional[dict] = dict(type='SiLU'),
#             init_cfg: Optional[dict] = None,
#     ):
#         super().__init__(init_cfg)
#         out_channels = out_channels or in_channels
#         hidden_channels = int(in_channels * hidden_channels_scale)
#
#         self.ffn_layers = nn.Sequential(
#             BCHW2BHWC(),
#             nn.LayerNorm(in_channels),
#             BHWC2BCHW(),
#             ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
#                        norm_cfg=norm_cfg, act_cfg=act_cfg),
#             ConvModule(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
#                        padding=hidden_kernel_size // 2, groups=hidden_channels,
#                        norm_cfg=norm_cfg, act_cfg=None),
#             GSiLU(),
#             nn.Dropout(dropout_rate),
#             ConvModule(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0,
#                        norm_cfg=norm_cfg, act_cfg=act_cfg),
#             nn.Dropout(dropout_rate),
#         )
#         self.add_identity = add_identity
#
#     def forward(self, x):
#         x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
#         return x
#
#
# class Stem(BaseModule):
#     """Stem layer"""
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             expansion: float = 1.0,
#             norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
#             act_cfg: Optional[dict] = dict(type='SiLU'),
#             init_cfg: Optional[dict] = None,
#     ):
#         super().__init__(init_cfg)
#         hidden_channels = make_divisible(int(out_channels * expansion), 8)
#
#         self.down_conv = ConvModule(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1,
#                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
#         self.conv1 = ConvModule(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1,
#                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
#         self.conv2 = ConvModule(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1,
#                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
#
#     def forward(self, x):
#         return self.conv2(self.conv1(self.down_conv(x)))
#
#
# class DownSamplingLayer(BaseModule):
#     """Down sampling layer"""
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: Optional[int] = None,
#             norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
#             act_cfg: Optional[dict] = dict(type='SiLU'),
#             init_cfg: Optional[dict] = None,
#     ):
#         super().__init__(init_cfg)
#         out_channels = out_channels or (in_channels * 2)
#
#         self.down_conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
#                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
#
#     def forward(self, x):
#         return self.down_conv(x)
#
#
# class InceptionBottleneck(BaseModule):
#     """Bottleneck with Inception module"""
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: Optional[int] = None,
#             kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
#             dilations: Sequence[int] = (1, 1, 1, 1, 1),
#             expansion: float = 1.0,
#             add_identity: bool = True,
#             with_caa: bool = True,
#             caa_kernel_size: int = 11,
#             norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
#             act_cfg: Optional[dict] = dict(type='SiLU'),
#             init_cfg: Optional[dict] = None,
#     ):
#         super().__init__(init_cfg)
#         out_channels = out_channels or in_channels
#         hidden_channels = make_divisible(int(out_channels * expansion), 8)
#
#         self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1,
#                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
#
#         self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
#                                   autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
#                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
#         self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
#                                    autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
#                                    groups=hidden_channels, norm_cfg=None, act_cfg=None)
#         self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
#                                    autopad(kernel_sizes[2], None, dilations[2]), dilations[2],
#                                    groups=hidden_channels, norm_cfg=None, act_cfg=None)
#         self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
#                                    autopad(kernel_sizes[3], None, dilations[3]), dilations[3],
#                                    groups=hidden_channels, norm_cfg=None, act_cfg=None)
#         self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[4], 1,
#                                    autopad(kernel_sizes[4], None, dilations[4]), dilations[4],
#                                    groups=hidden_channels, norm_cfg=None, act_cfg=None)
#         self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1,
#                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
#
#         if with_caa:
#             self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, None, None)
#         else:
#             self.caa_factor = None
#
#         self.add_identity = add_identity and in_channels == out_channels
#
#         self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
#                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
#
#     @autocast(True)
#     def forward(self, x):
#         x = self.pre_conv(x)
#
#         y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x
#         x = self.dw_conv(x)
#         x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
#         x = self.pw_conv(x)
#         if self.caa_factor is not None:
#             y = self.caa_factor(y)
#         if self.add_identity:
#             y = x * y
#             x = x + y
#         else:
#             x = x * y
#
#         x = self.post_conv(x)
#         return x
#
#
# class PKIBlock(BaseModule):
#     """Poly Kernel Inception Block"""
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: Optional[int] = None,
#             kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
#             dilations: Sequence[int] = (1, 1, 1, 1, 1),
#             with_caa: bool = True,
#             caa_kernel_size: int = 11,
#             expansion: float = 1.0,
#             ffn_scale: float = 4.0,
#             ffn_kernel_size: int = 3,
#             dropout_rate: float = 0.,
#             drop_path_rate: float = 0.,
#             layer_scale: Optional[float] = 1.0,
#             add_identity: bool = True,
#             norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
#             act_cfg: Optional[dict] = dict(type='SiLU'),
#             init_cfg: Optional[dict] = None,
#     ):
#         super().__init__(init_cfg)
#         out_channels = out_channels or in_channels
#         hidden_channels = make_divisible(int(out_channels * expansion), 8)
#
#         if norm_cfg is not None:
#             self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
#             self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]
#         else:
#             self.norm1 = nn.BatchNorm2d(in_channels)
#             self.norm2 = nn.BatchNorm2d(hidden_channels)
#
#         self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
#                                          expansion=1.0, add_identity=True,
#                                          with_caa=with_caa, caa_kernel_size=caa_kernel_size,
#                                          norm_cfg=norm_cfg, act_cfg=act_cfg)
#         self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False,
#                            norm_cfg=None, act_cfg=None)
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
#
#         self.layer_scale = layer_scale
#         if self.layer_scale:
#             self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
#             self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
#         self.add_identity = add_identity and in_channels == out_channels
#
#     def forward(self, x):
#         if self.layer_scale:
#             if self.add_identity:
#                 x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
#                 x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
#             else:
#                 x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
#                 x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
#         else:
#             if self.add_identity:
#                 x = x + self.drop_path(self.block(self.norm1(x)))
#                 x = x + self.drop_path(self.ffn(self.norm2(x)))
#             else:
#                 x = self.drop_path(self.block(self.norm1(x)))
#                 x = self.drop_path(self.ffn(self.norm2(x)))
#         return x


# _____________________________________ C3k2_PKIBlock ___________________________#
class Bottleneck_PKIBlock(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = PKIBlock(c_, c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_PKIBlock(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_PKIBlock(c_, c_) for _ in range(n)))


class C3k2_PKIBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_PKIBlock(self.c, self.c) if c3k else Bottleneck_iRMB(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_PKIBlock----------------------------------
class C3_PKIBlock(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_PKIBlock(c_, c_) for _ in range(n)))


# -------------------------C2f_PKIBlock----------------------------------
class C2f_PKIBlock(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_PKIBlock(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_PKIBlock ___________________________#
class Rep2ABlock_PKIBlock(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = PKIBlock(c_, c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_PKIBlock(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_PKIBlock(self.c, self.c) for _ in range(n))


# # -------------------------AdaptiveDilatedConv----------------------------------
#
# def generate_laplacian_pyramid(input_tensor, num_levels, size_align=True, mode='bilinear'):
#     """"
#     a alternative way for feature frequency decompose
#     """
#     pyramid = []
#     current_tensor = input_tensor
#     _, _, H, W = current_tensor.shape
#     for _ in range(num_levels):
#         b, _, h, w = current_tensor.shape
#         downsampled_tensor = F.interpolate(current_tensor, (h // 2 + h % 2, w // 2 + w % 2), mode=mode,
#                                            align_corners=(H % 2) == 1)  # antialias=True
#         if size_align:
#             # upsampled_tensor = F.interpolate(downsampled_tensor, (h, w), mode='bilinear', align_corners=(H%2) == 1)
#             # laplacian = current_tensor - upsampled_tensor
#             # laplacian = F.interpolate(laplacian, (H, W), mode='bilinear', align_corners=(H%2) == 1)
#             upsampled_tensor = F.interpolate(downsampled_tensor, (H, W), mode=mode, align_corners=(H % 2) == 1)
#             laplacian = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=(H % 2) == 1) - upsampled_tensor
#             # print(laplacian.shape)
#         else:
#             upsampled_tensor = F.interpolate(downsampled_tensor, (h, w), mode=mode, align_corners=(H % 2) == 1)
#             laplacian = current_tensor - upsampled_tensor
#         pyramid.append(laplacian)
#         current_tensor = downsampled_tensor
#     if size_align: current_tensor = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=(H % 2) == 1)
#     pyramid.append(current_tensor)
#     return pyramid
#
#
# class FrequencySelection(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  k_list=[2],
#                  # freq_list=[2, 3, 5, 7, 9, 11],
#                  lowfreq_att=True,
#                  fs_feat='feat',
#                  lp_type='freq',
#                  act='sigmoid',
#                  spatial='conv',
#                  spatial_group=1,
#                  spatial_kernel=3,
#                  init='zero',
#                  global_selection=False,
#                  ):
#         super().__init__()
#         # k_list.sort()
#         # print()
#         self.k_list = k_list
#         # self.freq_list = freq_list
#         self.lp_list = nn.ModuleList()
#         self.freq_weight_conv_list = nn.ModuleList()
#         self.fs_feat = fs_feat
#         self.lp_type = lp_type
#         self.in_channels = in_channels
#         # self.residual = residual
#         if spatial_group > 64: spatial_group = in_channels
#         self.spatial_group = spatial_group
#         self.lowfreq_att = lowfreq_att
#         if spatial == 'conv':
#             self.freq_weight_conv_list = nn.ModuleList()
#             _n = len(k_list)
#             if lowfreq_att:  _n += 1
#             for i in range(_n):
#                 freq_weight_conv = nn.Conv2d(in_channels=in_channels,
#                                              out_channels=self.spatial_group,
#                                              stride=1,
#                                              kernel_size=spatial_kernel,
#                                              groups=self.spatial_group,
#                                              padding=spatial_kernel // 2,
#                                              bias=True)
#                 if init == 'zero':
#                     freq_weight_conv.weight.data.zero_()
#                     freq_weight_conv.bias.data.zero_()
#                 else:
#                     # raise NotImplementedError
#                     pass
#                 self.freq_weight_conv_list.append(freq_weight_conv)
#         else:
#             raise NotImplementedError
#
#         if self.lp_type == 'avgpool':
#             for k in k_list:
#                 self.lp_list.append(nn.Sequential(
#                     nn.ReplicationPad2d(padding=k // 2),
#                     # nn.ZeroPad2d(padding= k // 2),
#                     nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
#                 ))
#         elif self.lp_type == 'laplacian':
#             pass
#         elif self.lp_type == 'freq':
#             pass
#         else:
#             raise NotImplementedError
#
#         self.act = act
#         # self.freq_weight_conv_list.append(nn.Conv2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 1, kernel_size=1, padding=0, bias=True))
#         self.global_selection = global_selection
#         if self.global_selection:
#             self.global_selection_conv_real = nn.Conv2d(in_channels=in_channels,
#                                                         out_channels=self.spatial_group,
#                                                         stride=1,
#                                                         kernel_size=1,
#                                                         groups=self.spatial_group,
#                                                         padding=0,
#                                                         bias=True)
#             self.global_selection_conv_imag = nn.Conv2d(in_channels=in_channels,
#                                                         out_channels=self.spatial_group,
#                                                         stride=1,
#                                                         kernel_size=1,
#                                                         groups=self.spatial_group,
#                                                         padding=0,
#                                                         bias=True)
#             if init == 'zero':
#                 self.global_selection_conv_real.weight.data.zero_()
#                 self.global_selection_conv_real.bias.data.zero_()
#                 self.global_selection_conv_imag.weight.data.zero_()
#                 self.global_selection_conv_imag.bias.data.zero_()
#
#     def sp_act(self, freq_weight):
#         if self.act == 'sigmoid':
#             freq_weight = freq_weight.sigmoid() * 2
#         elif self.act == 'softmax':
#             freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
#         else:
#             raise NotImplementedError
#         return freq_weight
#
#     def forward(self, x, att_feat=None):
#         """
#         att_feat:feat for gen att
#         """
#         # freq_weight = self.freq_weight_conv(x)
#         # self.sp_act(freq_weight)
#         # if self.residual: x_residual = x.clone()
#         if att_feat is None: att_feat = x
#         x_list = []
#         if self.lp_type == 'avgpool':
#             # for avg, freq_weight in zip(self.avg_list, self.freq_weight_conv_list):
#             pre_x = x
#             b, _, h, w = x.shape
#             for idx, avg in enumerate(self.lp_list):
#                 low_part = avg(x)
#                 high_part = pre_x - low_part
#                 pre_x = low_part
#                 # x_list.append(freq_weight[:, idx:idx+1] * high_part)
#                 freq_weight = self.freq_weight_conv_list[idx](att_feat)
#                 freq_weight = self.sp_act(freq_weight)
#                 # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
#                 tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group,
#                                                                                                -1, h, w)
#                 x_list.append(tmp.reshape(b, -1, h, w))
#             if self.lowfreq_att:
#                 freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
#                 # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
#                 tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h,
#                                                                                            w)
#                 x_list.append(tmp.reshape(b, -1, h, w))
#             else:
#                 x_list.append(pre_x)
#         elif self.lp_type == 'laplacian':
#             # for avg, freq_weight in zip(self.avg_list, self.freq_weight_conv_list):
#             # pre_x = x
#             b, _, h, w = x.shape
#             pyramids = generate_laplacian_pyramid(x, len(self.k_list), size_align=True)
#             # print('pyramids', len(pyramids))
#             for idx, avg in enumerate(self.k_list):
#                 # print(idx)
#                 high_part = pyramids[idx]
#                 freq_weight = self.freq_weight_conv_list[idx](att_feat)
#                 freq_weight = self.sp_act(freq_weight)
#                 # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
#                 tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group,
#                                                                                                -1, h, w)
#                 x_list.append(tmp.reshape(b, -1, h, w))
#             if self.lowfreq_att:
#                 freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
#                 # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
#                 tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pyramids[-1].reshape(b, self.spatial_group,
#                                                                                                   -1, h, w)
#                 x_list.append(tmp.reshape(b, -1, h, w))
#             else:
#                 x_list.append(pyramids[-1])
#         elif self.lp_type == 'freq':
#             pre_x = x.clone()
#             b, _, h, w = x.shape
#             # b, _c, h, w = freq_weight.shape
#             # freq_weight = freq_weight.reshape(b, self.spatial_group, -1, h, w)
#             x_fft = torch.fft.fftshift(torch.fft.fft2(x.double(), norm='ortho'))
#             # x_fft.to(torch.float32)
#             if self.global_selection:
#                 # global_att_real = self.global_selection_conv_real(x_fft.real)
#                 # global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, h, w)
#                 # global_att_imag = self.global_selection_conv_imag(x_fft.imag)
#                 # global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, h, w)
#                 # x_fft = x_fft.reshape(b, self.spatial_group, -1, h, w)
#                 # x_fft.real *= global_att_real
#                 # x_fft.imag *= global_att_imag
#                 # x_fft = x_fft.reshape(b, -1, h, w)
#                 # 将x_fft复数拆分成实部和虚部
#                 x_real = x_fft.real
#                 x_imag = x_fft.imag
#                 # 计算实部的全局注意力
#                 global_att_real = self.global_selection_conv_real(x_real)
#                 global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, h, w)
#                 # 计算虚部的全局注意力
#                 global_att_imag = self.global_selection_conv_imag(x_imag)
#                 global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, h, w)
#                 # 重塑x_fft为形状为(b, self.spatial_group, -1, h, w)的张量
#                 x_real = x_real.reshape(b, self.spatial_group, -1, h, w)
#                 x_imag = x_imag.reshape(b, self.spatial_group, -1, h, w)
#                 # 分别应用实部和虚部的全局注意力
#                 x_fft_real_updated = x_real * global_att_real
#                 x_fft_imag_updated = x_imag * global_att_imag
#                 # 合并为复数
#                 x_fft_updated = torch.complex(x_fft_real_updated, x_fft_imag_updated)
#                 # 重塑x_fft为形状为(b, -1, h, w)的张量
#                 x_fft = x_fft_updated.reshape(b, -1, h, w)
#
#             for idx, freq in enumerate(self.k_list):
#                 mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
#                 mask[:, :, round(h / 2 - h / (2 * freq)):round(h / 2 + h / (2 * freq)),
#                 round(w / 2 - w / (2 * freq)):round(w / 2 + w / (2 * freq))] = 1.0
#                 low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask), norm='ortho').real
#                 high_part = pre_x - low_part
#                 pre_x = low_part
#                 freq_weight = self.freq_weight_conv_list[idx](att_feat)
#                 freq_weight = self.sp_act(freq_weight)
#                 # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
#                 tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group,
#                                                                                                -1, h, w)
#                 x_list.append(tmp.reshape(b, -1, h, w))
#             if self.lowfreq_att:
#                 freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
#                 # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
#                 tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h,
#                                                                                            w)
#                 x_list.append(tmp.reshape(b, -1, h, w))
#             else:
#                 x_list.append(pre_x)
#         x = sum(x_list)
#         return x
#
#
# class AdaptiveDilatedConv(ModulatedDeformConv2d):
#     """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
#     layers.
#
#     Args:
#         in_channels (int): Same as nn.Conv2d.
#         out_channels (int): Same as nn.Conv2d.
#         kernel_size (int or tuple[int]): Same as nn.Conv2d.
#         stride (int): Same as nn.Conv2d, while tuple is not supported.
#         padding (int): Same as nn.Conv2d, while tuple is not supported.
#         dilation (int): Same as nn.Conv2d, while tuple is not supported.
#         groups (int): Same as nn.Conv2d.
#         bias (bool or str): If specified as `auto`, it will be decided by the
#             norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
#             False.
#     """
#
#     _version = 2
#
#     def __init__(self, *args,
#                  offset_freq=None,  # deprecated
#                  padding_mode='repeat',
#                  kernel_decompose='both',
#                  conv_type='conv',
#                  sp_att=False,
#                  pre_fs=True,  # False, use dilation
#                  epsilon=1e-4,
#                  use_zero_dilation=False,
#                  use_dct=False,
#                  fs_cfg={
#                      'k_list': [2, 4, 8],
#                      'fs_feat': 'feat',
#                      'lowfreq_att': False,
#                      'lp_type': 'freq',
#                      # 'lp_type':'laplacian',
#                      'act': 'sigmoid',
#                      'spatial': 'conv',
#                      'spatial_group': 1,
#                  },
#                  **kwargs):
#         super().__init__(*args, **kwargs)
#         if padding_mode == 'zero':
#             self.PAD = nn.ZeroPad2d(self.kernel_size[0] // 2)
#         elif padding_mode == 'repeat':
#             self.PAD = nn.ReplicationPad2d(self.kernel_size[0] // 2)
#         else:
#             self.PAD = nn.Identity()
#
#         self.kernel_decompose = kernel_decompose
#         self.use_dct = use_dct
#
#         if kernel_decompose == 'both':
#             self.OMNI_ATT1 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
#                                            groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
#             self.OMNI_ATT2 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels,
#                                            kernel_size=self.kernel_size[0] if self.use_dct else 1, groups=1,
#                                            reduction=0.0625, kernel_num=1, min_channel=16)
#         elif kernel_decompose == 'high':
#             self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
#                                           groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
#         elif kernel_decompose == 'low':
#             self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
#                                           groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
#         self.conv_type = conv_type
#         if conv_type == 'conv':
#             self.conv_offset = nn.Conv2d(
#                 self.in_channels,
#                 self.deform_groups * 1,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
#                 dilation=1,
#                 bias=True)
#
#         self.conv_mask = nn.Conv2d(
#             self.in_channels,
#             self.deform_groups * 1 * self.kernel_size[0] * self.kernel_size[1],
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
#             dilation=1,
#             bias=True)
#         if sp_att:
#             self.conv_mask_mean_level = nn.Conv2d(
#                 self.in_channels,
#                 self.deform_groups * 1,
#                 kernel_size=self.kernel_size,
#                 stride=self.stride,
#                 padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
#                 dilation=1,
#                 bias=True)
#
#         self.offset_freq = offset_freq
#
#         if self.offset_freq in ('FLC_high', 'FLC_res'):
#             self.LP = FLC_Pooling(freq_thres=min(0.5 * 1 / self.dilation[0], 0.25))
#         elif self.offset_freq in ('SLP_high', 'SLP_res'):
#             self.LP = StaticLP(self.in_channels, kernel_size=3, stride=1, padding=1, alpha=8)
#         elif self.offset_freq is None:
#             pass
#         else:
#             raise NotImplementedError
#
#         # An offset is like [y0, x0, y1, x1, y2, x2, ⋯, y8, x8]
#         offset = [-1, -1, -1, 0, -1, 1,
#                   0, -1, 0, 0, 0, 1,
#                   1, -1, 1, 0, 1, 1]
#         offset = torch.Tensor(offset)
#         # offset[0::2] *= self.dilation[0]
#         # offset[1::2] *= self.dilation[1]
#         # a tuple of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension
#         self.register_buffer('dilated_offset', torch.Tensor(offset[None, None, ..., None, None]))  # B, G, 18, 1, 1
#         if fs_cfg is not None:
#             if pre_fs:
#                 self.FS = FrequencySelection(self.in_channels, **fs_cfg)
#             else:
#                 self.FS = FrequencySelection(1, **fs_cfg)  # use dilation
#         self.pre_fs = pre_fs
#         self.epsilon = epsilon
#         self.use_zero_dilation = use_zero_dilation
#         self.init_weights()
#
#     def freq_select(self, x):
#         if self.offset_freq is None:
#             res = x
#         elif self.offset_freq in ('FLC_high', 'SLP_high'):
#             res = x - self.LP(x)
#         elif self.offset_freq in ('FLC_res', 'SLP_res'):
#             res = 2 * x - self.LP(x)
#         else:
#             raise NotImplementedError
#         return res
#
#     def init_weights(self):
#         super().init_weights()
#         if hasattr(self, 'conv_offset'):
#             # if isinstanace(self.conv_offset, nn.Conv2d):
#             if self.conv_type == 'conv':
#                 self.conv_offset.weight.data.zero_()
#                 # self.conv_offset.bias.data.fill_((self.dilation[0] - 1) / self.dilation[0] + 1e-4)
#                 self.conv_offset.bias.data.fill_((self.dilation[0] - 1) / self.dilation[0] + self.epsilon)
#             # self.conv_offset.bias.data.zero_()
#         # if hasattr(self, 'conv_offset'):
#         # self.conv_offset_low[1].weight.data.zero_()
#         # if hasattr(self, 'conv_offset_high'):
#         # self.conv_offset_high[1].weight.data.zero_()
#         # self.conv_offset_high[1].bias.data.zero_()
#         if hasattr(self, 'conv_mask'):
#             self.conv_mask.weight.data.zero_()
#             self.conv_mask.bias.data.zero_()
#
#         if hasattr(self, 'conv_mask_mean_level'):
#             self.conv_mask.weight.data.zero_()
#             self.conv_mask.bias.data.zero_()
#
#     # @force_fp32(apply_to=('x',))
#     # @force_fp32
#     def forward(self, x):
#         x_type = x.dtype
#         # offset = self.conv_offset(self.freq_select(x)) + self.conv_offset_low(self.freq_select(x))
#         if hasattr(self, 'FS') and self.pre_fs: x = self.FS(x)
#         # x = x.to(torch.float32)
#         if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
#             c_att1, f_att1, _, _, = self.OMNI_ATT1(x)
#             c_att2, f_att2, spatial_att2, _, = self.OMNI_ATT2(x)
#         elif hasattr(self, 'OMNI_ATT'):
#             c_att, f_att, _, _, = self.OMNI_ATT(x)
#
#         if self.conv_type == 'conv':
#             self.conv_offset.to(x.dtype)
#             offset = self.conv_offset(self.PAD(self.freq_select(x)))
#         elif self.conv_type == 'multifreqband':
#             self.conv_offset.to(x.dtype)
#             offset = self.conv_offset(self.freq_select(x))
#         # high_gate = self.conv_offset_high(x)
#         # high_gate = torch.exp(-0.5 * high_gate ** 2)
#         # offset = F.relu(offset, inplace=True) * self.dilation[0] - 1 # ensure > 0
#         if self.use_zero_dilation:
#             offset = (F.relu(offset + 1, inplace=True) - 1) * self.dilation[0]  # ensure > 0
#         else:
#             # offset = F.relu(offset, inplace=True) * self.dilation[0] # ensure > 0
#             offset = offset.abs() * self.dilation[0]  # ensure > 0
#             # offset[offset<0] = offset[offset<0].exp() - 1
#         # print(offset.mean(), offset.std(), offset.max(), offset.min())
#         if hasattr(self, 'FS') and (self.pre_fs == False): x = self.FS(x, F.interpolate(offset, x.shape[-2:],
#                                                                                         mode='bilinear', align_corners=(
#                                                                                                                                    x.shape[
#                                                                                                                                        -1] % 2) == 1))
#         # print(offset.max(), offset.abs().min(), offset.abs().mean())
#         # offset *= high_gate # ensure > 0
#         b, _, h, w = offset.shape
#         offset = offset.reshape(b, self.deform_groups, -1, h, w) * self.dilated_offset
#         # offset = offset.reshape(b, self.deform_groups, -1, h, w).repeat(1, 1, 9, 1, 1)
#         # offset[:, :, 0::2, ] *= self.dilated_offset[:, :, 0::2, ]
#         # offset[:, :, 1::2, ] *= self.dilated_offset[:, :, 1::2, ]
#         offset = offset.reshape(b, -1, h, w)
#
#         x = self.PAD(x)
#         self.conv_mask.to(x.dtype)
#         mask = self.conv_mask(x)
#         mask = mask.sigmoid()
#         # print(mask.shape)
#         # mask = mask.reshape(b, self.deform_groups, -1, h, w).softmax(dim=2)
#         if hasattr(self, 'conv_mask_mean_level'):
#             mask_mean_level = torch.sigmoid(self.conv_mask_mean_level(x)).reshape(b, self.deform_groups, -1, h, w)
#             mask = mask * mask_mean_level
#         mask = mask.reshape(b, -1, h, w)
#
#         if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
#             offset = offset.reshape(1, -1, h, w)
#             mask = mask.reshape(1, -1, h, w)
#             x = x.reshape(1, -1, x.size(-2), x.size(-1))
#             adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # b, c_out, c_in, k, k
#             adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
#             adaptive_weight_res = adaptive_weight - adaptive_weight_mean
#             _, c_out, c_in, k, k = adaptive_weight.shape
#             if self.use_dct:
#                 dct_coefficients = dct.dct_2d(adaptive_weight_res)
#                 # print(adaptive_weight_res.shape, dct_coefficients.shape)
#                 spatial_att2 = spatial_att2.reshape(b, 1, 1, k, k)
#                 dct_coefficients = dct_coefficients * (spatial_att2 * 2)
#                 # print(dct_coefficients.shape)
#                 adaptive_weight_res = dct.idct_2d(dct_coefficients)
#                 # adaptive_weight_res = adaptive_weight_res.reshape(b, c_out, c_in, k, k)
#                 # print(adaptive_weight_res.shape, dct_coefficients.shape)
#             # adaptive_weight = adaptive_weight_mean * (2 * c_att.unsqueeze(1)) * (2 * f_att.unsqueeze(2)) + adaptive_weight - adaptive_weight_mean
#             # adaptive_weight = adaptive_weight_mean * (c_att1.unsqueeze(1) * 2) * (f_att1.unsqueeze(2) * 2) + (adaptive_weight - adaptive_weight_mean) * (c_att2.unsqueeze(1) * 2) * (f_att2.unsqueeze(2) * 2)
#             adaptive_weight = adaptive_weight_mean * (c_att1.unsqueeze(1) * 2) * (
#                         f_att1.unsqueeze(2) * 2) + adaptive_weight_res * (c_att2.unsqueeze(1) * 2) * (
#                                           f_att2.unsqueeze(2) * 2)
#             adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
#             if self.bias is not None:
#                 bias = self.bias.repeat(b)
#             else:
#                 bias = self.bias
#             # print(adaptive_weight.shape)
#             # print(bias.shape)
#             # print(x.shape)
#             x = modulated_deform_conv2d(x, offset, mask, adaptive_weight, bias,
#                                         self.stride,
#                                         (self.kernel_size[0] // 2, self.kernel_size[1] // 2) if isinstance(self.PAD,
#                                                                                                            nn.Identity) else (
#                                         0, 0),  # padding
#                                         (1, 1),  # dilation
#                                         self.groups * b, self.deform_groups * b)
#         elif hasattr(self, 'OMNI_ATT'):
#             offset = offset.reshape(1, -1, h, w)
#             mask = mask.reshape(1, -1, h, w)
#             x = x.reshape(1, -1, x.size(-2), x.size(-1))
#             adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # b, c_out, c_in, k, k
#             adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
#             # adaptive_weight = adaptive_weight_mean * (2 * c_att.unsqueeze(1)) * (2 * f_att.unsqueeze(2)) + adaptive_weight - adaptive_weight_mean
#             if self.kernel_decompose == 'high':
#                 adaptive_weight = adaptive_weight_mean + (adaptive_weight - adaptive_weight_mean) * (
#                             c_att.unsqueeze(1) * 2) * (f_att.unsqueeze(2) * 2)
#             elif self.kernel_decompose == 'low':
#                 adaptive_weight = adaptive_weight_mean * (c_att.unsqueeze(1) * 2) * (f_att.unsqueeze(2) * 2) + (
#                             adaptive_weight - adaptive_weight_mean)
#
#             adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
#             # adaptive_bias = self.unsqueeze(0).repeat(b, 1, 1, 1, 1)
#             # print(adaptive_weight.shape)
#             # print(offset.shape)
#             # print(mask.shape)
#             # print(x.shape)
#             x = modulated_deform_conv2d(x, offset, mask, adaptive_weight, self.bias,
#                                         self.stride,
#                                         (self.kernel_size[0] // 2, self.kernel_size[1] // 2) if isinstance(self.PAD,
#                                                                                                            nn.Identity) else (
#                                         0, 0),  # padding
#                                         (1, 1),  # dilation
#                                         self.groups * b, self.deform_groups * b)
#         else:
#             x = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
#                                         self.stride,
#                                         (self.kernel_size[0] // 2, self.kernel_size[1] // 2) if isinstance(self.PAD,
#                                                                                                            nn.Identity) else (
#                                         0, 0),  # padding
#                                         (1, 1),  # dilation
#                                         self.groups, self.deform_groups)
#         # x = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
#         #                                self.stride, self.padding,
#         #                                self.dilation, self.groups,
#         #                                self.deform_groups)
#         # if hasattr(self, 'OMNI_ATT'): x = x * f_att
#         return x.reshape(b, -1, h, w).to(x_type)

# _____________________________________ FFCA ___________________________#
class Conv_withoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))


class SCAM(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(SCAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.k = Conv(in_channels, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = Conv_withoutBN(self.inter_channels, in_channels, 1, 1)
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

    def forward(self, x):
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        # avg max: [N, C, 1, 1]
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)

        # y2:[N, 1, H, W]
        y_avg = torch.matmul(avg, v).view(n, 1, h, w)
        y_max = torch.matmul(max, v).view(n, 1, h, w)

        # y_cat:[N, 2, H, W]
        y_cat = torch.cat((y_avg, y_max), 1)

        y = self.m(y) * self.m2(y_cat).sigmoid()

        return x + y


class FFM_Concat2(nn.Module):
    def __init__(self, dimension=1, Channel1=1, Channel2=1):
        super(FFM_Concat2, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel_all = int(Channel1 + Channel2)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型 parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()

        w = self.w[:(C1 + C2)]  # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x = [x1, x2]
        return torch.cat(x, self.d)


class FFM_Concat3(nn.Module):
    def __init__(self, dimension=1, Channel1=1, Channel2=1, Channel3=1):
        super(FFM_Concat3, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel3 = Channel3
        self.Channel_all = int(Channel1 + Channel2 + Channel3)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()
        N3, C3, H3, W3 = x[2].size()

        w = self.w[:(C1 + C2 + C3)]  # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:(C1 + C2)] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x3 = (weight[(C1 + C2):] * x[2].view(N3, H3, W3, C3)).view(N3, C3, H3, W3)
        x = [x1, x2, x3]
        return torch.cat(x, self.d)


class BasicConv_FFCA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv_FFCA, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv_FFCA(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv_FFCA(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv_FFCA(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv_FFCA(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv_FFCA((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv_FFCA(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv_FFCA(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv_FFCA(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


# _____________________________________ DEABlock ___________________________#
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class DEABlock(nn.Module):
    def __init__(self, dim, kernel_size=5, reduction=8):
        super(DEABlock, self).__init__()
        self.conv1 = Conv(dim, dim, kernel_size)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = Conv(dim, dim, kernel_size)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res


class DEBlock(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super(DEBlock, self).__init__()
        self.conv1 = Conv(dim, dim, kernel_size)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = Conv(dim, dim, kernel_size)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res


# _____________________________________ C3k2_DEABlock ___________________________#
class Bottleneck_DEABlock(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = DEABlock(c_)
        self.cv3 = DEBlock(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_DEABlock(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_DEABlock(c_, c_) for _ in range(n)))


class C3k2_DEABlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_DEABlock(self.c, self.c) if c3k else Bottleneck_DEABlock(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_DEABlock----------------------------------
class C3_DEABlock(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DEABlock(c_, c_) for _ in range(n)))


# -------------------------C2f_DEABlock----------------------------------
class C2f_DEABlock(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_DEABlock(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_DEABlock ___________________________#
class Rep2ABlock_DEABlock(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = DEABlock(c_)
        self.cv3 = DEBlock(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_DEABlock(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_DEABlock(self.c, self.c) for _ in range(n))


# _____________________________________ SKFusion  ___________________________#
class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


# _____________________________________ C3k2_gnconv ___________________________#
class Bottleneck_gnconv(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = gnconv(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_gnconv(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_gnconv(c_, c_) for _ in range(n)))


class C3k2_gnconv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_gnconv(self.c, self.c) if c3k else Bottleneck_gnconv(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_gnconv----------------------------------
class C3_gnconv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_gnconv(c_, c_) for _ in range(n)))


# -------------------------C2f_gnconv----------------------------------
class C2f_gnconv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_gnconv(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_gnconv ___________________________#
class Rep2ABlock_gnconv(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels

        self.cv3 = gnconv(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_gnconv(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_gnconv(self.c, self.c) for _ in range(n))


# _____________________________________ RCAB ___________________________#
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=1, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(Conv(n_feat, n_feat, kernel_size))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# _____________________________________ C3k2_RCAB ___________________________#
class Bottleneck_RCAB(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = RCAB(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_RCAB(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_RCAB(c_, c_) for _ in range(n)))


class C3k2_RCAB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_RCAB(self.c, self.c) if c3k else Bottleneck_RCAB(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_RCAB----------------------------------
class C3_RCAB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RCAB(c_, c_) for _ in range(n)))


# -------------------------C2f_RCAB----------------------------------
class C2f_RCAB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_RCAB(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_RCAB ___________________________#
class Rep2ABlock_RCAB(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = RCAB(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_RCAB(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_RCAB(self.c, self.c) for _ in range(n))


# _____________________________________ RCBv6 ___________________________#
class RCBv6(nn.Module):
    def __init__(self, n_feats, lk=4):
        super().__init__()
        self.LKA = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 5, 1, lk // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 7, stride=1, padding=9, groups=n_feats, dilation=3),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.Sigmoid())

        self.LFE = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1))

    def forward(self, x):
        shortcut = x.clone()
        x = self.LFE(x)

        x = self.LKA(x) * x

        return x + shortcut


# _____________________________________ C3k2_RCBv6 ___________________________#
class Bottleneck_RCBv6(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = RCBv6(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_RCBv6(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_RCBv6(c_, c_) for _ in range(n)))


class C3k2_RCBv6(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_RCBv6(self.c, self.c) if c3k else Bottleneck_RCBv6(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_RCBv6----------------------------------
class C3_RCBv6(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RCBv6(c_, c_) for _ in range(n)))


# -------------------------C2f_RCBv6----------------------------------
class C2f_RCBv6(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_RCBv6(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_RCBv6 ___________________________#
class Rep2ABlock_RCBv6(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = RCBv6(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_RCBv6(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_RCBv6(self.c, self.c) for _ in range(n))


# _____________________________________ MLKA_Ablation ___________________________#
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLKA_Ablation(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        k = 2

        # Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // k, n_feats // k, 7, 1, 7 // 2, groups=n_feats // k),
            nn.Conv2d(n_feats // k, n_feats // k, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // k, dilation=4),
            nn.Conv2d(n_feats // k, n_feats // k, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // k, n_feats // k, 5, 1, 5 // 2, groups=n_feats // k),
            nn.Conv2d(n_feats // k, n_feats // k, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // k, dilation=3),
            nn.Conv2d(n_feats // k, n_feats // k, 1, 1, 0))
        '''self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k),  
            nn.Conv2d(n_feats//k, n_feats//k, 5, stride=1, padding=(5//2)*2, groups=n_feats//k, dilation=2),
            nn.Conv2d(n_feats//k, n_feats//k, 1, 1, 0))'''

        # self.X3 = nn.Conv2d(n_feats//k, n_feats//k, 3, 1, 1, groups= n_feats//k)
        self.X5 = nn.Conv2d(n_feats // k, n_feats // k, 5, 1, 5 // 2, groups=n_feats // k)
        self.X7 = nn.Conv2d(n_feats // k, n_feats // k, 7, 1, 7 // 2, groups=n_feats // k)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()

        x = self.norm(x)

        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)

        # u_1, u_2, u_3= torch.chunk(u, 3, dim=1)
        a_1, a_2 = torch.chunk(a, 2, dim=1)

        a = torch.cat([self.LKA7(a_1) * self.X7(a_1), self.LKA5(a_2) * self.X5(a_2)], dim=1)

        x = self.proj_last(x * a) * self.scale + shortcut

        return x


# _____________________________________ C3k2_MLKA_Ablation ___________________________#
class Bottleneck_MLKA_Ablation(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = MLKA_Ablation(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_MLKA_Ablation(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_MLKA_Ablation(c_, c_) for _ in range(n)))


class C3k2_MLKA_Ablation(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_MLKA_Ablation(self.c, self.c) if c3k else Bottleneck_MLKA_Ablation(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_MLKA_Ablation----------------------------------
class C3_MLKA_Ablation(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_MLKA_Ablation(c_, c_) for _ in range(n)))


# -------------------------C2f_MLKA_Ablation----------------------------------
class C2f_MLKA_Ablation(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_MLKA_Ablation(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_MLKA_Ablation ___________________________#
class Rep2ABlock_MLKA_Ablation(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = MLKA_Ablation(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_MLKA_Ablation(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_MLKA_Ablation(self.c, self.c) for _ in range(n))


# _____________________________________ ConvMod ___________________________#
class LayerNorm_convmod(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # N, C, H, W = x.shape
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        x = a * v
        x = self.proj(x)
        return x


# _____________________________________ C3k2_ConvMod ___________________________#
class Bottleneck_ConvMod(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = ConvMod(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_ConvMod(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_ConvMod(c_, c_) for _ in range(n)))


class C3k2_ConvMod(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_ConvMod(self.c, self.c) if c3k else Bottleneck_ConvMod(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_ConvMod----------------------------------
class C3_ConvMod(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ConvMod(c_, c_) for _ in range(n)))


# -------------------------C2f_ConvMod----------------------------------
class C2f_ConvMod(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_ConvMod(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_ConvMod ___________________________#
class Rep2ABlock_ConvMod(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = ConvMod(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_ConvMod(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_ConvMod(self.c, self.c) for _ in range(n))


# _____________________________________ PagFM ___________________________#
class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels[0], mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels[1], mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels[0],
                          kernel_size=1, bias=False),
                BatchNorm(in_channels[0])
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_size = x[0].size()
        if self.after_relu:
            y = self.relu(x[1])
            x = self.relu(x[0])
        else:
            y = x[1]
            x = x[0]

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y

        return x


# _____________________________________ WCMF ___________________________#
class WCMF(nn.Module):
    def __init__(self, channel=256):
        super(WCMF, self).__init__()
        self.conv_r1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_d1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())

        self.conv_c1 = nn.Sequential(nn.Conv2d(2*channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, 2, 3, 1, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def fusion(self,f1,f2,f_vec):

        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2

    def forward(self,x):
        rgb = x[0]
        depth = x[1]
        Fr = self.conv_r1(rgb)
        Fd = self.conv_d1(depth)
        f = torch.cat([Fr, Fd],dim=1)
        f = self.conv_c1(f)
        f = self.conv_c2(f)
        # f = self.avgpool(f)
        Fo = self.fusion(Fr, Fd, f)
        return Fo


# _____________________________________ C3k2_WTConv2d ___________________________#
class Bottleneck_WTConv2d(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = WTConv2d(c_, c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_WTConv2d(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_WTConv2d(c_, c_) for _ in range(n)))


class C3k2_WTConv2d(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_WTConv2d(self.c, self.c) if c3k else Bottleneck_WTConv2d(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_WTConv2d----------------------------------
class C3_WTConv2d(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_WTConv2d(c_, c_) for _ in range(n)))


# -------------------------C2f_WTConv2d----------------------------------
class C2f_WTConv2d(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_WTConv2d(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_WTConv2d ___________________________#
class Rep2ABlock_WTConv2d(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = WTConv2d(c_, c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_WTConv2d(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_WTConv2d(self.c, self.c) for _ in range(n))


# _____________________________________ ConvolutionalGLU ___________________________#
class ConvolutionalGLU(nn.Module):
    # def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    def __init__(self, c1, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.c1 = c1
        self.fc1 = nn.Linear(c1, c1)
        self.dwconv = DWConv(c1, c1)
        self.act = act_layer()
        self.fc2 = nn.Linear(c1, c1)
        self.drop = nn.Dropout(drop)

        # return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
    def forward(self, x):
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        x = self.fc1(p).permute(1, 2, 0).reshape(b, self.c1, w, h)
        x, v = x.chunk(2, dim=-1)
        x = torch.cat([self.act(self.dwconv(x)), v], -1)
        x = self.drop(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        x = self.fc2(p).permute(1, 2, 0).reshape(b, self.c1, w, h)
        # x = self.fc2(x)
        x = self.drop(x)
        return x


# _____________________________________ C3k2_ConvolutionalGLU ___________________________#
class Bottleneck_ConvolutionalGLU(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = ConvolutionalGLU(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_ConvolutionalGLU(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_ConvolutionalGLU(c_, c_) for _ in range(n)))


class C3k2_ConvolutionalGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_ConvolutionalGLU(self.c, self.c) if c3k else Bottleneck_ConvolutionalGLU(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_ConvolutionalGLU----------------------------------
class C3_ConvolutionalGLU(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ConvolutionalGLU(c_, c_) for _ in range(n)))


# -------------------------C2f_ConvolutionalGLU----------------------------------
class C2f_ConvolutionalGLU(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_ConvolutionalGLU(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_ConvolutionalGLU ___________________________#
class Rep2ABlock_ConvolutionalGLU(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = ConvolutionalGLU(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_ConvolutionalGLU(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_ConvolutionalGLU(self.c, self.c) for _ in range(n))


# _____________________________________ RCM ___________________________#

class ConvMlp(nn.Module):
    """ 使用 1x1 卷积保持空间维度的 MLP
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# rectangular self-calibration attention (RCA)
class RCA(nn.Module):
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            nn.Sigmoid()
        )

    def sge(self, x):
        # [N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w  # .repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather)  # [N, 1, C, 1]

        return ge

    def forward(self, x):
        loc = self.dwconv_hw(x)
        att = self.sge(x)
        out = att * loc

        return out


# Rectangular Self-Calibration Module (RCM)
class RCM(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer=RCA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size,
                                       ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


# _____________________________________ C3k2_RCM ___________________________#
class Bottleneck_RCM(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = RCM(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_RCM(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_RCM(c_, c_) for _ in range(n)))


class C3k2_RCM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_RCM(self.c, self.c) if c3k else Bottleneck_RCM(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_RCM----------------------------------
class C3_RCM(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RCM(c_, c_) for _ in range(n)))


# -------------------------C2f_RCM----------------------------------
class C2f_RCM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_RCM(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_RCM ___________________________#
class Rep2ABlock_RCM(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = RCM(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_RCM(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_RCM(self.c, self.c) for _ in range(n))


# _____________________________________ EUCB ___________________________#
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class EUCB(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = c1
        self.out_channels = c2
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


# _____________________________________ MSDC ___________________________#
class MSDC(nn.Module):
    def __init__(self, in_channels, stride=1, kernel_sizes=[3, 5], activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        return torch.cat(outputs, dim=1)


# _____________________________________ MSCB ___________________________#
class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB)
    """

    def __init__(self, in_channels, out_channels, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.stride, self.kernel_sizes, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(3*self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = msdc_outs
            # for dwout in msdc_outs:
            #     dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, self.gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

    def gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a


# _____________________________________ LGAG ___________________________#
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        F_int = F_g + F_l
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

    def forward(self, x):
        g, x = x[0], x[1]
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi

# _____________________________________ start_Block ___________________________#
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


# _____________________________________ ds_ ___________________________#
class ds_(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, c1, c2, ):
        super(ds_, self).__init__()
        c_ = c2//2
        self.conv = Conv(c1, c_, 2, 2, p=0)
        self.conv_mid = Conv(c1*4, c2-c_, 1, 1)

    def forward(self, x):
        y = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        y = self.conv_mid(y)
        return torch.cat([y, self.conv(x)], dim=1)


# _____________________________________ FFCM ___________________________#
class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1,3,5,7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num
        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(  # DW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

        self.bn = torch.nn.BatchNorm2d(dim*2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))

        return x


class FFCM(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_gloal=Freq_Fusion,
            mixer_kernel_size=[1,3,5,7],
            local_size=8
    ):
        super(FFCM, self).__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size,
                                 se_ratio=8, local_size=local_size)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local_1 = self.dw_conv_1(x[0])
        x_local_2 = self.dw_conv_2(x[0])
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1))
        x = self.ca_conv(x_gloal)
        x = self.ca(x) * x

        return x


# _____________________________________ C3k2_FFCM ___________________________#
class Bottleneck_FFCM(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = FFCM(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_FFCM(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_FFCM(c_, c_) for _ in range(n)))


class C3k2_FFCM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_FFCM(self.c, self.c) if c3k else Bottleneck_FFCM(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_FFCM----------------------------------
class C3_FFCM(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_FFCM(c_, c_) for _ in range(n)))


# -------------------------C2f_FFCM----------------------------------
class C2f_FFCM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_FFCM(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_FFCM ___________________________#
class Rep2ABlock_FFCM(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = FFCM(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_FFCM(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_FFCM(self.c, self.c) for _ in range(n))


# _____________________________________ FSAS ___________________________#
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class FSAS_LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(FSAS_LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FSAS(nn.Module):
    def __init__(self, dim, bias=True):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = FSAS_LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


# _____________________________________ C3k2_FSAS ___________________________#
class Bottleneck_FSAS(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = FSAS(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_FSAS(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_FSAS(c_, c_) for _ in range(n)))


class C3k2_FSAS(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_FSAS(self.c, self.c) if c3k else Bottleneck_FSAS(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_FSAS----------------------------------
class C3_FSAS(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_FSAS(c_, c_) for _ in range(n)))


# -------------------------C2f_FSAS----------------------------------
class C2f_FSAS(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_FSAS(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_FSAS ___________________________#
class Rep2ABlock_FSAS(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = FSAS(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_FSAS(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_FSAS(self.c, self.c) for _ in range(n))


# _____________________________________ SDI ___________________________#
class Ghost_SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [GhostConv(i, channel[-1], k=3, s=1) for i in channel])

    def forward(self, xs):
        ans = torch.ones_like(xs[-1])
        target_size = xs[-1].shape[2:]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear',  align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(i, channel[-1], kernel_size=3, stride=1, padding=1) for i in channel])

    def forward(self, xs):
        ans = torch.ones_like(xs[-1])
        target_size = xs[-1].shape[2:]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear',  align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


# -------------------------Gold YOLO----------------------------------
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x


def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool


class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                0] != out_channels else nn.Identity()
        self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                1] != out_channels else nn.Identity()
        self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[
                                                                                2] != out_channels else nn.Identity()
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)

        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])

        x0 = self.cv1(self.downsample(x[0], output_size))
        x1 = self.cv2(x[1])
        x2 = self.cv3(F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False))
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])

        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d

        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out


class IFM(nn.Module):
    def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(ouc))
        )

    def forward(self, x):
        return self.conv(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_inp: list,
            flag: int
    ) -> None:
        super().__init__()
        self.global_inp = global_inp
        self.flag = flag
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(global_inp[self.flag], oup, 1, act=False)
        self.global_act = Conv(global_inp[self.flag], oup, 1, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        '''
        x_g: global features
        x_l: local features
        '''
        x_l, x_g = x
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        gloabl_info = x_g.split(self.global_inp, dim=1)[self.flag]

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(gloabl_info)
        global_feat = self.global_embedding(gloabl_info)

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class PyramidPoolAgg(nn.Module):
    def __init__(self, inc, ouc, stride, pool_mode='torch'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
        self.conv = Conv(inc, ouc)

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return self.conv(torch.cat(out, dim=1))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv(in_features, hidden_features, act=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.ReLU6()
        self.fc2 = Conv(hidden_features, out_features, act=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GOLDYOLO_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv(dim, nh_kd, 1, act=False)
        self.to_k = Conv(dim, nh_kd, 1, act=False)
        self.to_v = Conv(dim, self.dh, 1, act=False)

        self.proj = torch.nn.Sequential(nn.ReLU6(), Conv(self.dh, dim, act=False))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class top_Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = GOLDYOLO_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    def __init__(self, embedding_dim, ouc_list, block_num=2, key_dim=8, num_heads=4,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path))
        self.conv = nn.Conv2d(embedding_dim, sum(ouc_list), 1)

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return self.conv(x)


class AdvPoolFusion(nn.Module):
    def forward(self, x):
        x1, x2 = x
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d

        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)

        return torch.cat([x1, x2], 1)


# _____________________________________ C3k2_LSK ___________________________#
class Bottleneck_LSK(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = LSKblock(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_LSK(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_LSK(c_, c_) for _ in range(n)))


class C3k2_LSK(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_LSK(self.c, self.c) if c3k else Bottleneck_LSK(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_LSKblock ----------------------------------
class C3_LSK(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_LSK(c_, c_) for _ in range(n)))


# -------------------------C2f_LSK----------------------------------
class C2f_LSK(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_LSK(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_LSK ___________________________#
class Rep2ABlock_LSK(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = LSKblock(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_LSK(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_LSK(self.c, self.c) for _ in range(n))


# _____________________________________ SFH_former_Block ___________________________#
class TokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(TokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim//2

        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)

        return x


class TokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(TokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x+x0)

        return x


class Mixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=TokenMixer_For_Local,
            token_mixer_for_gloal=TokenMixer_For_Gloal,
    ):
        super(Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim,)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim,)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*dim, 2*dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim//2, 2*dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2*dim, 1),
        )



    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)



        return x


class SFH_former_FFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(SFH_former_FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)


        return x


class SFH_former_Block(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=Mixer,
    ):
        super(SFH_former_Block, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim)
        self.ffn = SFH_former_FFN(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy

        return x


# _____________________________________ C3k2_SFH_former_Block ___________________________#
class Bottleneck_SFH_former_Block(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = SFH_former_Block(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_SFH_former_Block(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_SFH_former_Block(c_, c_) for _ in range(n)))


class C3k2_SFH_former_Block(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_SFH_former_Block(self.c, self.c) if c3k else Bottleneck_SFH_former_Block(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_SFH_former_Block----------------------------------
class C3_SFH_former_Block(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_SFH_former_Block(c_, c_) for _ in range(n)))


# -------------------------C2f_SFH_former_Block----------------------------------
class C2f_SFH_former_Block(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_SFH_former_Block(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_SFH_former_Block ___________________________#
class Rep2ABlock_SFH_former_Block(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = SFH_former_Block(c_)

    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_SFH_former_Block(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_SFH_former_Block(self.c, self.c) for _ in range(n))


# _____________________________________ BiFPN ___________________________#
class BiFPN(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN, self).__init__()
        self.w = nn.Parameter(torch.ones(len(c1), dtype=torch.float32), requires_grad=True)
        self.conv = Conv(sum(c1), c2, k=1, s=1, p=0)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(torch.cat([weight[i] * x[i] for i in range(len(x))], dim=1))



# ————————————————————————————————AFPN————————————————————————————————
class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, 5, 4, 2)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


def BasicConv(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("silu", nn.SiLU(inplace=True)),
    ]))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.silu(out)

        return out




class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True )
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class ScaleBlockBody(nn.Module):
    def __init__(self, channels=[128, 256, 512]):
        super(ScaleBlockBody, self).__init__()

        self.blocks_top1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_mid1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_bot1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )

        self.downsample_top1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_mid1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_top1 = ASFF_2(inter_dim=channels[0])
        self.asff_mid1 = ASFF_2(inter_dim=channels[1])

        self.blocks_top2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0])
        )
        self.blocks_mid2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1])
        )

        self.downsample_top2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_top2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_mid2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_mid2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_bot2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_bot2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_top2 = ASFF_3(inter_dim=channels[0])
        self.asff_mid2 = ASFF_3(inter_dim=channels[1])
        self.asff_bot2 = ASFF_3(inter_dim=channels[2])

        self.blocks_top3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0])
        )
        self.blocks_mid3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1])
        )
        self.blocks_bot3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2])
        )

    def forward(self, x):
        x1, x2, x3 = x

        x1 = self.blocks_top1(x1)
        x2 = self.blocks_mid1(x2)
        x3 = self.blocks_bot1(x3)

        top = self.asff_top1(x1, self.upsample_mid1_2(x2))
        mid = self.asff_mid1(self.downsample_top1_2(x1), x2)

        x1 = self.blocks_top2(top)
        x2 = self.blocks_mid2(mid)

        top = self.asff_top2(x1, self.upsample_mid2_2(x2), self.upsample_bot2_4(x3))
        mid = self.asff_mid2(self.downsample_top2_2(x1), x2, self.upsample_bot2_2(x3))
        bot = self.asff_bot2(self.downsample_top2_4(x1), self.downsample_mid2_2(x2), x3)

        top = self.blocks_top3(top)
        mid = self.blocks_mid3(mid)
        bot = self.blocks_bot3(bot)

        return top, mid, bot


class AFPN(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], out_channels=[256, 512, 1024]):
        super(AFPN, self).__init__()

        self.conv1 = BasicConv(in_channels[0], in_channels[0] // 4, 1)
        self.conv2 = BasicConv(in_channels[1], in_channels[1] // 4, 1)
        self.conv3 = BasicConv(in_channels[2], in_channels[2] // 4, 1)

        self.body = nn.Sequential(
            ScaleBlockBody([in_channels[0] // 4, in_channels[1] // 4, in_channels[2] // 4])
        )

        self.conv11 = BasicConv(in_channels[0] // 4, out_channels[0], 1)
        self.conv22 = BasicConv(in_channels[1] // 4, out_channels[1], 1)
        self.conv33 = BasicConv(in_channels[2] // 4, out_channels[2], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x1, x2, x3 = x

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        out1, out2, out3 = self.body([x1, x2, x3])

        out1 = self.conv11(out1)
        out2 = self.conv22(out2)
        out3 = self.conv33(out3)

        return tuple([out1, out2, out3])


class get_feturemap(nn.Module):
    def __init__(self, c2, c):
        super(get_feturemap, self).__init__()
        self.c = c

    def forward(self, x):
        return x[self.c]


## Multi-DConv Head Transposed Self-Attention (MDHTA)
class MDHTA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(MDHTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# _____________________________________ C3k2_MDHTA ___________________________#
class Bottleneck_MDHTA(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = MDHTA(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_MDHTA(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_MDHTA(c_, c_) for _ in range(n)))


class C3k2_MDHTA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_MDHTA(self.c, self.c) if c3k else Bottleneck_MDHTA(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_MDHTA ----------------------------------
class C3_MDHTA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_MDHTA(c_, c_) for _ in range(n)))


# -------------------------C2f_MDHTA ----------------------------------
class C2f_MDHTA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_MDHTA(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_MDHTA ___________________________#
class Rep2ABlock_MDHTA(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = MDHTA(c_)

    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_MDHTA(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_MDHTA(self.c, self.c) for _ in range(n))


# _____________________________________ ADown_light ___________________________#
class ADown_light(nn.Module):
    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = GhostConv(2 * c1, self.c, 1, 1)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 3, 1, 1, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.cat([x2[..., ::2, ::2], x2[..., 1::2, ::2], x2[..., ::2, 1::2], x2[..., 1::2, 1::2]], 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


# _____________________________________ ContrastDrivenFeatureAggregation: CDFA ___________________________#
class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class CDFA(nn.Module):
    "ContrastDrivenFeatureAggregation"
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5


        self.v = nn.Linear(dim, dim)

        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        fg, bg = x[1], x[2]
        x = self.input_cbr(x[0])


        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)


        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')


        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)


        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')


        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)


        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            CBR(256, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            CBR(256, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            CBR(256, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        f_uc = self.cbr_uc(x)
        return [f_fg, f_bg, f_uc]


class ConDSeg_model(nn.Module):
    def __init__(self, c1=1024, c2=256):
        super(ConDSeg_model, self).__init__()
        self.m1 = nn.Sequential(DecoupleLayer(c1, c2), CDFA(c2, c2, 2))

    def forward(self, x):
        x = self.m1(x)
        return x


# _____________________________________ CDFAPreprocess ___________________________#
class CDFAPreprocess(nn.Module):

    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x


# _____________________________________ MetaSeg ___________________________#
class ChannelReductionAttention(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        # self.dim2 = dim2
        self.pool_ratio = pool_ratio
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()


    def forward(self, x, h, w):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp_MetaSeg(nn.Module):
    """Mlp implemented by with 1*1 convolutions.
    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 #  act_cfg=dict(type='GELU'),
                 act_layer=nn.GELU,
                 drop_path=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        # self.act = build_activation_layer(act_cfg)
        self.act = act_layer()

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop_path)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class global_meta_block(nn.Module):

    def __init__(self, dim1, num_heads=2, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm3 = norm_layer(dim1)

        self.attn = ChannelReductionAttention(dim1=dim1, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp_MetaSeg(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_path=drop_path)

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm3(x), h, w))
        return x.permute(0, 2, 1).reshape(n, -1, h, w)


# _____________________________________ DilatedMDTA ___________________________#
class DilatedMDTA(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False):
        super(DilatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# _____________________________________ C3k2_DilatedMDTA ___________________________#
class Bottleneck_DilatedMDTA(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = DilatedMDTA(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_DilatedMDTA(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_DilatedMDTA(c_, c_) for _ in range(n)))


class C3k2_DilatedMDTA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_DilatedMDTA(self.c, self.c) if c3k else Bottleneck_DilatedMDTA(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_DilatedMDTA----------------------------------
class C3_DilatedMDTA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DilatedMDTA(c_, c_) for _ in range(n)))


# -------------------------C2f_DilatedMDTA----------------------------------
class C2f_DilatedMDTA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_DilatedMDTA(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_DilatedMDTA ___________________________#
class Rep2ABlock_DilatedMDTA(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = DilatedMDTA(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_DilatedMDTA(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_DilatedMDTA(self.c, self.c) for _ in range(n))


# _____________________________________ HighDctFrequencyExtractor ___________________________#
class HighDctFrequencyExtractor(nn.Module):
    def __init__(self, alpha=0.05):
        super(HighDctFrequencyExtractor, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None

    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)

        return torch.matmul(self.dct_matrix_h.to(x.device), torch.matmul(x, self.dct_matrix_w.t().to(x.device)))

    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)

        return torch.matmul(self.dct_matrix_h.t().to(x.device), torch.matmul(x, self.dct_matrix_w.to(x.device)))

    def high_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        mask[:alpha_h, :alpha_w] = 0

        return x * mask

    def forward(self, x):
        xq = self.dct_2d(x)
        xq_high = self.high_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals)
        return xh


# _____________________________________ LowDctFrequencyExtractor ___________________________#
class LowDctFrequencyExtractor(nn.Module):
    def __init__(self, alpha=0.95):
        super(LowDctFrequencyExtractor, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None

    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)

        return torch.matmul(self.dct_matrix_h.to(x.device), torch.matmul(x, self.dct_matrix_w.t().to(x.device)))

    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)

        return torch.matmul(self.dct_matrix_h.t().to(x.device), torch.matmul(x, self.dct_matrix_w.to(x.device)))

    def high_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        mask[-alpha_h:, -alpha_w:] = 0

        return x * mask

    def forward(self, x):
        xq = self.dct_2d(x)
        xq_high = self.high_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals)
        return xh

# _____________________________________ LHConcat ___________________________#
class LHConcat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.low = LowDctFrequencyExtractor()
        self.high = HighDctFrequencyExtractor()

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat([self.low(x[0]), self.high(x[1])], self.d)


# _____________________________________ SABlock ___________________________#
class SV_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def SV_block(x,block_size):
    B,H,W,C = x.shape
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B,Hp//block_size,block_size,Wp//block_size,block_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x, H, Hp, C

def unblock(x, Ho):
    B,H,W,win_H,win_W,C = x.shape
    x = x.permute(0,1,3,2,4,5).contiguous().reshape(B,H*win_H,W*win_W, C)
    Wp = Hp = H*win_H
    Wo = Ho
    if Hp > Ho or Wp > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x

def alter_sparse(x, sparse_size=2):
    x = x.permute(0, 2, 3, 1)
    assert x.shape[1]%sparse_size == 0 & x.shape[2]%sparse_size == 0, 'image size should be divisible by block_size'
    grid_size = x.shape[1]//sparse_size
    out, H, Hp, C = SV_block(x, grid_size)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, C)
    out = out.permute(0, 3, 1, 2)
    return out, H, Hp, C


def alter_unsparse(x, H, Hp, C, sparse_size=2):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, Hp//sparse_size, Hp//sparse_size, sparse_size, sparse_size, C)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = unblock(x, H)
    out = out.permute(0, 3, 1, 2)
    return out

class SVDWConv(nn.Module):
    def __init__(self, dim=768):
        super(SVDWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class SVMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SVDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads=2, sparse_size=2, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = SV_Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SVMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls = True
        self.sparse_size = sparse_size
        if self.ls:
            init_value = 1e-6
            # print(f"Use layer_scale, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x_befor = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape
        if self.ls:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            # x = x_befor + self.gamma_1 * x
            x = x + self.gamma_2 * self.mlp(self.norm2(x), H, W)
        else:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            # x = x_befor + x
            x = x + self.mlp(self.norm2(x), H, W)
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


# _____________________________________ C3_R-ELAN ___________________________#
class C3_R_ELAN(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(nn.Sequential(*(ABlock(c_, max(c_//32, 1), 2, 1) for _ in range(2))) for _ in range(n))

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


# _____________________________________ RCRep2A_R-ELAN ___________________________#
class Rep2ABlock_A2(nn.Module):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # mid channels
        self.cv1 = RepConv(c1, c_, 3)
        self.att1 = Attention_xy(c_)
        self.att2 = Attention_xy(c_)
        self.cv2 = RepConv(2 * c_, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.att1(self.cv1(x))
        y = torch.cat((y, self.att2(y)), 1)
        return x + self.cv2(y) if self.add else self.cv2(y)


class RCRep2A_A2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # self.cv_m = Conv(self.c, self.c, 3)
        self.cv_m = GhostConv(self.c, self.c, 3)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Rep2ABlock_A2(self.c, self.c, shortcut, e=e) for _ in range(n))

    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, 1)
        y = list([self.cv_m(x1), x1, x2])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# _____________________________________ GBS ___________________________#
class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)


class GBC(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)

        return x + residual


# _____________________________________ C3k2_GBC ___________________________#
class Bottleneck_GBC(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = GBC(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_GBC(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_GBC(c_, c_) for _ in range(n)))


class C3k2_GBC(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_GBC(self.c, self.c) if c3k else Bottleneck_GBC(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_GBC----------------------------------
class C3_GBC(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_GBC(c_, c_) for _ in range(n)))


# -------------------------C2f_GBC----------------------------------
class C2f_GBC(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_GBC(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_GBC ___________________________#
class Rep2ABlock_GBC(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = GBC(c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_GBC(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_GBC(self.c, self.c) for _ in range(n))


# _____________________________________ ARConv ___________________________#
class ARConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, l_max=9, w_max=9, flag=False, modulation=True):
        super(ARConv, self).__init__()
        self.lmax = l_max
        self.wmax = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.flag = flag
        self.modulation = modulation
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
                for i in self.i_list
            ]
        )
        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride)
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.Dropout2d(0),
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.hook_handles = []
        self.hook_handles.append(self.m_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.m_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[1].register_full_backward_hook(self._set_lr))

        self.reserved_NXY = nn.Parameter(torch.tensor([3, 3], dtype=torch.int32), requires_grad=False)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return grad_input

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()  # 移除钩子函数
        self.hook_handles.clear()  # 清空句柄列表

    def forward(self, x):
        # assert isinstance(hw_range, list) and len(
        #     hw_range) == 2, "hw_range should be a list with 2 elements, represent the range of h w"
        # scale = hw_range[1] // 9
        scale = 1
        # if hw_range[0] == 1 and hw_range[1] == 3:
        #     scale = 1
        m = self.m_conv(x)
        bias = self.b_conv(x)
        offset = self.p_conv(x * 100)
        l = self.l_conv(offset) * 1 + 1  # b, 1, h, w
        w = self.w_conv(offset) * 1 + 1  # b, 1, h, w
        mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)
        mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)
        N_X = int(torch.div(mean_l, scale, rounding_mode='trunc'))
        N_Y = int(torch.div(mean_w, scale, rounding_mode='trunc'))

        def phi(x):
            if x % 2 == 0:
                x -= 1
            return x

        N_X, N_Y = phi(N_X), phi(N_Y)
        N_X, N_Y = max(N_X, 3), max(N_Y, 3)
        N_X, N_Y = min(N_X, 7), min(N_Y, 7)

        # self.reserved_NXY = self.reserved_NXY = nn.Parameter(
        #     torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device),
        #     requires_grad=False
        # )
        # else:
        #     N_X = self.reserved_NXY[0]
        #     N_Y = self.reserved_NXY[1]

        N = N_X * N_Y
        # print(N_X, N_Y)
        l = l.repeat([1, N, 1, 1])
        w = w.repeat([1, N, 1, 1])
        offset = torch.cat((l, w), dim=1)
        dtype = offset.data.type()
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype, N_X, N_Y)  # (b, 2*N, h, w)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2*N)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # clip p
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # (b, c, h, w, N)
        x_offset = (
                g_lt.unsqueeze(dim=1) * x_q_lt
                + g_rb.unsqueeze(dim=1) * x_q_rb
                + g_lb.unsqueeze(dim=1) * x_q_lb
                + g_rt.unsqueeze(dim=1) * x_q_rt
        )
        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
        x_offset = self.dropout2(x_offset)
        x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)
        out = x_offset * m + bias
        return out

    def _get_p_n(self, N, dtype, n_x, n_y):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
            indexing = 'ij'
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing='ij'
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype, n_x, n_y):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        L, W = offset.split([N, N], dim=1)
        L = L / n_x
        W = W / n_y
        offsett = torch.cat([L, W], dim=1)
        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat([1, 1, h, w])
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offsett * p_n
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, n_x, n_y):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + n_y].contiguous().view(b, c, h, w * n_y) for s in range(0, N, n_y)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
        return x_offset


# _____________________________________ DynamicTanh ___________________________#
class DynamicTanh(nn.Module):
    def __init__(self, c1, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = c1
        self.alpha_init_value = alpha_init_value
        self.channels_last = c1

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(c1))
        self.bias = nn.Parameter(torch.zeros(c1))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            # x = x * self.weight + self.bias
            x = (x.permute(0, 2, 3, 1) * self.weight  + self.bias).permute(0, 3, 1, 2)
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

# _____________________________________ Conv_DyT ___________________________#
class Conv_DyT(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = DynamicTanh(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


# _____________________________________ LCA ___________________________#
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        # self.norm = LayerNorm(dim)
        self.norm = nn.BatchNorm2d(dim)
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x):
        x = x + self.ffn(self.norm(x), self.norm(x))
        x = self.gdfn(self.norm(x))
        return x


class I_LCA(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False):
        super(I_LCA, self).__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)

    def forward(self, x):
        x = x + self.ffn(self.norm(x), self.norm(x))
        x = x + self.gdfn(self.norm(x))
        return x


# _____________________________________ C3k2_HV_LCA ___________________________#
class Bottleneck_HV_LCA(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = HV_LCA(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_HV_LCA(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_HV_LCA(c_, c_) for _ in range(n)))


class C3k2_HV_LCA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_HV_LCA(self.c, self.c) if c3k else Bottleneck_HV_LCA(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_HV_LCA ----------------------------------
class C3_HV_LCA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_HV_LCA(c_, c_) for _ in range(n)))


# -------------------------C2f_HV_LCA----------------------------------
class C2f_HV_LCA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_HV_LCA(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_HV_LCA ___________________________#
class Rep2ABlock_HV_LCA(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = HV_LCA(c_)

    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_HV_LCA(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_HV_LCA(self.c, self.c) for _ in range(n))


# _____________________________________ MuLUTUnit ___________________________#
class ActConv(nn.Module):
    """ Conv. with activation. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))

class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, in_nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out


class MuLUTUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, c1, c2,  mode='1x1', dense=True):
        super().__init__()
        self.act = nn.ReLU()
        # self.upscale = upscale
        c_ = c2//4
        if mode == '2x2':
            self.conv1 = Conv(c1, c_, 2)
        elif mode == '2x2d':
            self.conv1 = Conv(c1, c_, 2, dilation=2)
        elif mode == '2x2d3':
            self.conv1 = Conv(c1, c_, 2, dilation=3)
        elif mode == '1x4':
            self.conv1 = Conv(c1, c_, (1, 4))
        elif mode == '1x3':
            self.conv1 = Conv(c1, c_ (1, 3))
        elif mode == '1x1':
            self.conv1 = Conv(c1, c_, (1, 1))
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(c_)
            self.conv3 = DenseConv(c_ * 2)
            self.conv4 = DenseConv(c_ * 4)
            self.conv5 = DenseConv(c_ * 8)
            if mode == '1x1':
                self.conv6 = Conv(c_ * 16, c2, 1)
            else:
                self.conv6 = Conv(c_ * 16, c2, 1)
        else:
            self.conv2 = ActConv(c2, c2, 1)
            self.conv3 = ActConv(c2, c2, 1)
            self.conv4 = ActConv(c2, c2, 1)
            self.conv5 = ActConv(c2, c2, 1)
            if mode == '1x1':
                self.conv6 = Conv(c2, c2, 3)
            else:
                self.conv6 = Conv(c2, c2, 1)
        # if self.upscale > 1:
        #     self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        # if self.upscale > 1:
        #     x = self.pixel_shuffle(x)
        return x


# _____________________________________ C3k2_MuLUTUnit ___________________________#
class Bottleneck_MuLUTUnit(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = MuLUTUnit(c_, c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_MuLUTUnit(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_MuLUTUnit(c_, c_) for _ in range(n)))


class C3k2_MuLUTUnit(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_MuLUTUnit(self.c, self.c) if c3k else Bottleneck_MuLUTUnit(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_MuLUTUnit----------------------------------
class C3_MuLUTUnit(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_MuLUTUnit(c_, c_) for _ in range(n)))


# -------------------------C2f_MuLUTUnit----------------------------------
class C2f_MuLUTUnit(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_MuLUTUnit(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_MuLUTUnit ___________________________#
class Rep2ABlock_MuLUTUnit(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = MuLUTUnit(c_,c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_MuLUTUnit(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_MuLUTUnit(self.c, self.c) for _ in range(n))


# _____________________________________ DBlock & EBlock ___________________________#
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d_DBlock(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d_DBlock, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)  # the dconv
        )

    def forward(self, input):
        return self.branch(input)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class DBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c,
                                    bias=True, dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand=1, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d_DBlock(c)
        self.norm2 = LayerNorm2d_DBlock(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, adapter=None):

        y = inp
        x = self.norm1(inp)
        # x = self.conv1(self.extra_conv(x))
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # second step
        x = self.conv4(self.norm2(y))  # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x)  # size [B, C, H, W]
        x = y + x * self.gamma

        return x


class FreMLP(nn.Module):

    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        # second step

        self.norm1 = LayerNorm2d_DBlock(c)
        self.norm2 = LayerNorm2d_DBlock(c)
        self.freq = FreMLP(nc=c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def forward(self, inp):
        y = inp
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # second step
        x_step2 = self.norm2(y)  # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2)  # size [B, C, H, W]
        x = y * x_freq
        x = y + x * self.gamma
        return x


# _____________________________________ token_mixer ___________________________#
class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn,1,1]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2
        return out


class token_mixer(nn.Module):
    def __init__(self, in_chn,dilation1=1,dilation2=2,dilation3=4, kernel_size=3, stride=1, padding='same'):
        super(token_mixer, self).__init__()
        self.move = LearnableBiasnn(in_chn)
        self.cov1 = Conv(in_chn, in_chn, kernel_size, stride, d=dilation1)
        self.cov2 = Conv(in_chn, in_chn, kernel_size, stride, d=dilation2)
        self.cov3 = Conv(in_chn, in_chn, kernel_size, stride, d=dilation3)
        self.norm = nn.LayerNorm(in_chn)
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn)
        self.act3 = RPReLU(in_chn)


    def forward(self, x):
        B,C,H,W = x.shape
        x = self.move(x)
        x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1,2)
        x1 = self.act1(x1)
        x2 = self.cov2(x).permute(0, 2, 3, 1).flatten(1,2)
        x2 = self.act2(x2)
        x3 = self.cov3(x).permute(0, 2, 3, 1).flatten(1,2)
        x3 = self.act3(x3)
        x = self.norm(x1+x2+x3)
        return x.permute(0, 2, 1).view(-1, C, H, W).contiguous()


# _____________________________________ C3k2_token_mixer ___________________________#
class Bottleneck_token_mixer(Bottleneck):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv3 = token_mixer(c_)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x) + self.cv3(self.cv1(x))) if self.add else self.cv2(self.cv1(x) + self.cv3(self.cv1(x)))


class C3k_token_mixer(C3k):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_token_mixer(c_, c_) for _ in range(n)))


class C3k2_token_mixer(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(
            C3k_token_mixer(self.c, self.c) if c3k else Bottleneck_token_mixer(self.c, self.c) for _ in range(n)
        )


# -------------------------C3_token_mixer----------------------------------
class C3_token_mixer(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_token_mixer(c_, c_) for _ in range(n)))


# -------------------------C2f_token_mixer----------------------------------
class C2f_token_mixer(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(Bottleneck_token_mixer(self.c, self.c) for _ in range(n))


# _____________________________________ RCRep2A_token_mixer ___________________________#
class Rep2ABlock_token_mixer(Rep2ABlock):
    """Standard block by xy."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__(c1, c2, shortcut, e)
        c_ = int(c2 * e)  # mid channels
        self.cv3 = model(c_, c_)


    def forward(self, x):
        z = self.cv1(x)
        return x + self.cv2(self.att(z+self.cv3(z))) if self.add else self.cv2(self.att(z+self.cv3(z)))


class RCRep2A_token_mixer(RCRep2A):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Rep2ABlock_token_mixer(self.c, self.c) for _ in range(n))


# _____________________________________ LRSA ___________________________#
class LRSA_Attention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v)
        "此方法需要pytorch版本2.0及以上"
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        output (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


class LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim=32, mlp_dim=64, heads=1):
        super().__init__()

        self.layer = nn.ModuleList([
            PreNorm(dim, LRSA_Attention(dim, heads, qk_dim)),
            PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x):
        ps = 6
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)

        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)

        return x


# _____________________________________ PConv2 ___________________________#
# Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection
class PConv2(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


# _____________________________________ OverLoCK ___________________________#
def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (
    kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print(
                '---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (
                conv_bias - bn.running_mean) * bn.weight / std


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim)
    )


def downsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
    )


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x


class LayerNorm2d_OverLoCK(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """

    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(self, channels, kernel_size=5, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class CTXDownsample(nn.Module):
    def __init__(self, dim, h_dim=8):
        super().__init__()

        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim)
        )
        self.h_proj = nn.Sequential(
            nn.Conv2d(h_dim // 4, h_dim // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim // 4)
        )

    def forward(self, x, ctx):
        x = self.x_proj(x)
        ctx = self.h_proj(ctx)
        return (x, ctx)


class ResDWConv(nn.Conv2d):
    '''
    Depthwise convolution with residual connection
    '''

    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        x = x + super().forward(x)
        return x


class RepConvBlock(nn.Module):

    def __init__(self,
                 dim=64,
                 kernel_size=7,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d_OverLoCK,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False):
        super().__init__()

        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint

        mlp_dim = int(dim * mlp_ratio)

        self.dwconv = ResDWConv(dim, kernel_size=3)

        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False,
                                attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )

        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward_features(self, x):

        x = self.dwconv(x)

        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        return x

    def forward(self, x):

        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)

        return x


# class DynamicConvBlock(nn.Module):
#     def __init__(self,
#                  dim=64,
#                  ctx_dim=32,
#                  kernel_size=7,
#                  smk_size=5,
#                  num_heads=2,
#                  mlp_ratio=4,
#                  ls_init_value=None,
#                  res_scale=False,
#                  drop_path=0,
#                  norm_layer=LayerNorm2d_OverLoCK,
#                  is_first=False,
#                  is_last=False,
#                  use_gemm=False,
#                  deploy=False,
#                  use_checkpoint=False,
#                  **kwargs):
#
#         super().__init__()
#
#         ctx_dim = ctx_dim // 4
#         out_dim = dim + ctx_dim
#         mlp_dim = int(dim * mlp_ratio)
#         self.kernel_size = kernel_size
#         self.res_scale = res_scale
#         self.use_gemm = use_gemm
#         self.smk_size = smk_size
#         self.num_heads = num_heads * 2
#         head_dim = dim // self.num_heads
#         self.scale = head_dim ** -0.5
#         self.is_first = is_first
#         self.is_last = is_last
#         self.use_checkpoint = use_checkpoint
#
#         if not is_first:
#             self.x_scale = LayerScale(ctx_dim, init_value=1)
#             self.h_scale = LayerScale(ctx_dim, init_value=1)
#
#         self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
#         self.norm1 = norm_layer(out_dim)
#
#         self.fusion = nn.Sequential(
#             nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
#             nn.BatchNorm2d(out_dim),
#             nn.GELU(),
#             nn.Conv2d(out_dim, dim, kernel_size=1),
#             GRN(dim),
#         )
#
#         self.weight_query = nn.Sequential(
#             nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim // 2),
#         )
#
#         self.weight_key = nn.Sequential(
#             nn.AdaptiveAvgPool2d(7),
#             nn.Conv2d(ctx_dim, dim // 2, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim // 2),
#         )
#
#         self.weight_proj = nn.Conv2d(49, kernel_size ** 2 + smk_size ** 2, kernel_size=1)
#
#         self.dyconv_proj = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim),
#         )
#
#         self.lepe = nn.Sequential(
#             DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False,
#                                 attempt_use_lk_impl=use_gemm),
#             nn.BatchNorm2d(dim),
#         )
#
#         self.se_layer = SEModule(dim)
#
#         self.gate = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.SiLU(),
#         )
#
#         self.proj = nn.Sequential(
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, out_dim, kernel_size=1),
#         )
#
#         self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
#         self.norm2 = norm_layer(out_dim)
#
#         self.mlp = nn.Sequential(
#             nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
#             nn.GELU(),
#             ResDWConv(mlp_dim, kernel_size=3),
#             GRN(mlp_dim),
#             nn.Conv2d(mlp_dim, out_dim, kernel_size=1),
#         )
#
#         self.ls1 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
#         self.ls2 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
#         self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
#
#         self.get_rpb()
#
#     def get_rpb(self):
#         self.rpb_size1 = 2 * self.smk_size - 1
#         self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
#         self.rpb_size2 = 2 * self.kernel_size - 1
#         self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
#         nn.init.zeros_(self.rpb1)
#         nn.init.zeros_(self.rpb2)
#
#     @torch.no_grad()
#     def generate_idx(self, kernel_size):
#         rpb_size = 2 * kernel_size - 1
#         idx_h = torch.arange(0, kernel_size)
#         idx_w = torch.arange(0, kernel_size)
#         idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
#         return (idx_h, idx_w, idx_k)
#
#     def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
#         """
#         RPB implementation directly borrowed from https://tinyurl.com/mrbub4t3
#         """
#         num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
#         num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
#         num_repeat_h[kernel_size // 2] = height - (kernel_size - 1)
#         num_repeat_w[kernel_size // 2] = width - (kernel_size - 1)
#         bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (
#                     2 * kernel_size - 1)) + idx_w.repeat_interleave(num_repeat_w)
#         bias_idx = bias_hw.unsqueeze(-1) + idx_k
#         bias_idx = bias_idx.reshape(-1, int(kernel_size ** 2))
#         bias_idx = torch.flip(bias_idx, [0])
#         rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
#         rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size ** 2))
#         return attn + rpb
#
#     def _forward_inner(self, x, h_x, h_r):
#         input_resoltion = x.shape[2:]
#         B, C, H, W = x.shape
#         B, C_h, H_h, W_h = h_x.shape
#
#         if not self.is_first:
#             h_x = self.x_scale(h_x) + self.h_scale(h_r)
#
#         x_f = torch.cat([x, h_x], dim=1)
#         x_f = self.dwconv1(x_f)
#         identity = x_f
#         x_f = self.norm1(x_f)
#         x = self.fusion(x_f)
#         gate = self.gate(x)
#         lepe = self.lepe(x)
#
#         is_pad = False
#         if min(H, W) < self.kernel_size:
#             is_pad = True
#             if H < W:
#                 size = (self.kernel_size, int(self.kernel_size / H * W))
#             else:
#                 size = (int(self.kernel_size / W * H), self.kernel_size)
#             x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
#             x_f = F.interpolate(x_f, size=size, mode='bilinear', align_corners=False)
#             H, W = size
#
#         query, key = torch.split(x_f, split_size_or_sections=[C, C_h], dim=1)
#         query = self.weight_query(query) * self.scale
#         key = self.weight_key(key)
#         query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
#         key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
#         weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
#         weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
#         weight = self.weight_proj(weight)
#         weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)
#
#         attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size ** 2, self.kernel_size ** 2], dim=-1)
#         rpb1_idx = self.generate_idx(self.smk_size)
#         rpb2_idx = self.generate_idx(self.kernel_size)
#         attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
#         attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
#         attn1 = torch.softmax(attn1, dim=-1)
#         attn2 = torch.softmax(attn2, dim=-1)
#         value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)
#
#         x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
#         x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)
#
#         x = torch.cat([x1, x2], dim=1)
#         x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)
#
#         if is_pad:
#             x = F.adaptive_avg_pool2d(x, input_resoltion)
#
#         x = self.dyconv_proj(x)
#
#         x = x + lepe
#         x = self.se_layer(x)
#
#         x = gate * x
#         x = self.proj(x)
#
#         if self.res_scale:
#             x = self.ls1(identity) + self.drop_path(x)
#         else:
#             x = identity + self.drop_path(self.ls1(x))
#
#         x = self.dwconv2(x)
#
#         if self.res_scale:
#             x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
#         else:
#             x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
#
#         if self.is_last:
#             return (x, None)
#         else:
#             l_x, h_x = torch.split(x, split_size_or_sections=[C, C_h], dim=1)
#             return (l_x, h_x)
#
#     def forward(self, x, h_x, h_r):
#         if self.use_checkpoint and x.requires_grad:
#             x = checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
#         else:
#             x = self._forward_inner(x, h_x, h_r)
#         return x


# _____________________________________ Octave ___________________________#
#Frequency-aware module
class FirstOctaveConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels),  # (512,256)
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride

        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_h2h + X_l2h
        return X_h


class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)

        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)  # 同频输入、输出
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)  # 不同频输入、输出

        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x0 = x
        x_h, x_l = self.fir(x)
        x_hh, x_ll = x_h, x_l,
        # x_1 = x_hh +x_ll
        x_h_1, x_l_1 = self.mid1((x_h, x_l))
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1))
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2))
        x_ret = self.lst((x_h_5, x_l_5))
        return x_ret


# A synergistic CNN-transformer network with pooling attention fusion for hyperspectral image classification
class HPA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(HPA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  #Y avg
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  #X avg
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))  #Y avg
        self.max_w = nn.AdaptiveMaxPool2d((1, None))  #X avg

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w  --->2048,2,11,11
        x_h = self.pool_h(group_x) #2048,2,11,1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) #2048,2,1,11--->2048,2,11,1
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) #2048,2,22,1
        x_h, x_w = torch.split(hw, [h, w], dim=2) #2048,2,11,1
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) #2048,2,11,11
        x2 = self.conv3x3(group_x)  #2048,2,11,11

        y_h = self.max_h(group_x) #2048,2,11,1
        y_w = self.max_w(group_x).permute(0, 1, 3, 2)
        yhw = self.conv1x1(torch.cat([y_h, y_w], dim=2)) #2048,2,22,1
        y_h, y_w = torch.split(yhw, [h, w], dim=2) #2048,2,11,1
        y1 = self.gn(group_x * y_h.sigmoid() * y_w.permute(0, 1, 3, 2).sigmoid()) #2048,2,11,11
        y11 = y1.reshape(b * self.groups, c // self.groups, -1) # b*g, c//g, hw 2048,2,121
        y12 = self.softmax(self.map(y1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) #2048,1,2

        x11 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw 2048,2,121
        x12 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) #2048,2,1,1-->2048,2,1--->2048,1,2
        x21 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw  #2048,2,121
        x22 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) #2048,2,1,1-->2048,2,1--->2048,1,2
        weights = (torch.matmul(x12, y11) + torch.matmul(y12, x11)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class TBFE(nn.Module):
    def __init__(self, input_channels, reduction_N=32):
        super(TBFE, self).__init__()
        self.point_wise = nn.Conv2d(input_channels, reduction_N, kernel_size=1, padding=0, bias=False)
        self.depth_wise = nn.Sequential(nn.Conv2d(reduction_N, reduction_N, kernel_size=(3, 3), padding=1),
                                        nn.BatchNorm2d(reduction_N), nn.ReLU(), )

        self.conv3D = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 3), padding=(0, 0, 1),
                                stride=(1, 1, 1), bias=False)
        self.bn = nn.BatchNorm2d(reduction_N)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.point_wise(x)
        x_2 = self.depth_wise(x_1)
        x_2 = x_1 + x_2

        # DSC
        x_3 = x_1.unsqueeze(1)
        x_3 = self.conv3D(x_3)
        x_3 = x_3.squeeze(1)
        x = torch.cat((x_2, x_3), dim=1)

        return x

    """LayerNorm for channels of 2D tensor(B C H W)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


# EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality
class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, bn_weight_init=0)

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x


class Stem(nn.Module):
    def __init__(self, in_dim=3, dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, dim // 8, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 2, dim, kernel_size=3, stride=2, padding=1, act_layer=None))

    def forward(self, x):
        x = self.conv(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=4.0):
        super().__init__()
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, hidden_dim, kernel_size=1),
            ConvLayer2D(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim),
            SqueezeExcite(hidden_dim, .25),
            ConvLayer2D(hidden_dim, out_dim, kernel_size=1, act_layer=None)
        )

        self.dwconv1 = ConvLayer2D(in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None)
        self.dwconv2 = ConvLayer2D(out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None)

    def forward(self, x):
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


class HSMSSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3 * state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)
        self.hz_proj = ConvLayer1D(d_model, 2 * self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x, H, W):
        batch, _, L = x.shape
        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, W)).flatten(2)
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)

        AB = (A * B)
        h = x @ AB.transpose(-2, -1)

        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z.clone()) + h * self.D)
        y = h @ C  # B C N, B C L -> B C L

        y = y.view(batch, -1, H, W).contiguous()  # + x * self.D  # B C H W
        return y, h


class EfficientViMBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)

        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)

        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))

        # LayerScale
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        # DWconv1
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        # HSM-SSD
        x_prev = x
        H, W = x.shape[2:]
        x, h = self.mixer(self.norm(x.flatten(2)), H, W)
        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        # DWConv2
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)

        # FFN
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)
        # return x, h
        return x


# class EfficientViMStage(nn.Module):
#     def __init__(self, in_dim, out_dim, depth=2, mlp_ratio=4., downsample=None, ssd_expand=1, state_dim=64):
#         super().__init__()
#         self.depth = depth
#         self.blocks = nn.ModuleList([
#             EfficientViMBlock(dim=in_dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand, state_dim=state_dim) for _ in
#             range(depth)])
#
#         self.downsample = downsample(in_dim=in_dim, out_dim=out_dim) if downsample is not None else None
#
#     def forward(self, x):
#         for blk in self.blocks:
#             x, h = blk(x)
#
#         x_out = x
#         if self.downsample is not None:
#             x = self.downsample(x)
#         # return x, x_out, h
#         return x


# Reciprocal Attention Mixing Transformer for Lightweight Image Restoration
# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
class MobiVari1(nn.Module):  # MobileNet v1 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride, kernel_size // 2, groups=dim)
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()

    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x)) + x))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * self.kernel_size * self.kernel_size * self.dim + H * W * 1 * 1 * self.dim * self.out_dim  # self.dw_conv + self.pw_conv
        return flops


class MobiVari2(MobiVari1):  # MobileNet v2 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None, exp_factor=1.2, expand_groups=4):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim * exp_factor)
        expand_dim = expand_dim + (expand_groups - expand_dim % expand_groups)
        self.expand_dim = expand_dim

        self.exp_conv = nn.Conv2d(dim, self.expand_dim, 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(expand_dim, expand_dim, kernel_size, stride, kernel_size // 2, groups=expand_dim)
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1) + x1))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * 1 * 1 * (self.dim // self.expand_groups) * self.expand_dim  # self.exp_conv
        flops += H * W * self.kernel_size * self.kernel_size * self.expand_dim  # self.dw_conv
        flops += H * W * 1 * 1 * self.expand_dim * self.out_dim  # self.pw_conv
        return flops


class HRAMi(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, mv_ver=1, mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(HRAMi, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        if mv_ver == 1:
            self.mobivari = MobiVari1(dim + dim // 4 + dim // 16 + dim, kernel_size, stride, act=mv_act, out_dim=dim)
        elif mv_ver == 2:
            self.mobivari = MobiVari2(dim + dim // 4 + dim // 16 + dim, kernel_size, stride, act=mv_act, out_dim=dim,
                                      exp_factor=2., expand_groups=1)

    def forward(self, attn_list):
        for i, attn in enumerate(attn_list[:-1]):
            attn = F.pixel_shuffle(attn, 2 ** i)
            x = attn if i == 0 else torch.cat([x, attn], dim=1)
        x = torch.cat([x, attn_list[-1]], dim=1)
        x = self.mobivari(x)
        return x

    def flops(self, resolutions):
        return self.mobivari.flops(resolutions)


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# https://arxiv.org/abs/2401.16456
class GroupNorm_SHViT(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN_SHViT(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear_SHViT(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""

    def __init__(self, dim, qk_dim=4, pdim=4):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm_SHViT(pdim)

        self.qkv = Conv2d_BN_SHViT(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN_SHViT(
            dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))

        return x


# # https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# # https://arxiv.org/pdf/2408.08345​
# class MonaOp(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
#         self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
#         self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
#
#         self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )
#
#     def forward(self, x):
#         identity = x
#         conv1_x = self.conv1(x)
#         conv2_x = self.conv2(x)
#         conv3_x = self.conv3(x)
#
#         x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
#
#         identity = x
#
#         x = self.projector(x)
#
#         return identity + x
#
# from mmengine.model import BaseModule
#
# class Mona(BaseModule):
#     def __init__(self, in_dim,):
#         super().__init__()
#
#         self.project1 = nn.Linear(in_dim, 64)
#         self.nonlinear = F.gelu
#         self.project2 = nn.Linear(64, in_dim)
#
#         self.dropout = nn.Dropout(p=0.1)
#
#         self.adapter_conv = MonaOp(64)
#
#         self.norm = nn.LayerNorm(in_dim)
#         self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
#         self.gammax = nn.Parameter(torch.ones(in_dim))
#
#     def forward(self, x):
#         # identity = x
#
#         B, C, h, w = x.shape
#         x = x.reshape(B, C, -1).permute(0, 2, 1)
#         x = self.norm(x) * self.gamma + x * self.gammax
#         identity = x
#
#         project1 = self.project1(x)
#
#         b, n, c = project1.shape
#         project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
#         project1 = self.adapter_conv(project1)
#         project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
#
#         nonlinear = self.nonlinear(project1)
#         nonlinear = self.dropout(nonlinear)
#         project2 = self.project2(nonlinear)
#
#         return (identity + project2).permute(0, 1, 2).reshape(B, C, h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm_SEFN(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_SEFN, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class FeedForward_SEF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super(FeedForward_SEF, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                            groups=hidden_features, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm_SEFN(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm_SEFN(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x_list):
        x, spatial = x_list
        x = self.project_in(x)
        #### Spatial branch
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)

        # Ensure x1 and y have the same spatial size before concatenation
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        # Resize y to match the spatial dimensions of x1
        y = F.interpolate(y, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=False)

        x1 = self.fusion(torch.cat((x1, y), dim=1))
        x1 = self.dwconv_afterfusion(x1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# https://ieeexplore.ieee.org/document/10786275
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)  # B 6C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


# 先空间校正再通道校正
class FeatureCorrection_s2c(nn.Module):
    def __init__(self, dim, reduction=1, eps=1e-8):
        super(FeatureCorrection_s2c, self).__init__()
        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        spatial_weights = self.spatial_weights(x1, x2)
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1

        channel_weights = self.channel_weights(x1_1, x2_1)

        main_out = x1_1 + fuse_weights[1] * channel_weights[1] * x2_1
        aux_out = x2_1 + fuse_weights[1] * channel_weights[0] * x1_1
        return torch.cat([main_out, aux_out], dim=1)


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# https://arxiv.org/abs/2412.20066
class SequenceShuffleAttention(nn.Module):
    def __init__(self, c1, hidden_features=None, group=4, act_layer=nn.GELU, input_resolution=(64,64)):
        super().__init__()
        self.group = group
        self.input_resolution = input_resolution
        self.in_features = c1
        self.out_features = c1

        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1, groups=self.group, kernel_size=1, stride=1, padding=0),  # 卷积层，使用分组卷积
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def channel_rearrange(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, self.group, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, x):
        y = x
        x = self.channel_shuffle(x)
        x = self.gating(x)
        x = self.channel_rearrange(x)
        return y * x


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# https://github.com/kkkls/EVSSM/tree/master
class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8

        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        return x


# https://arxiv.org/abs/2408.01897
# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
class ACFMAttention(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False):
        super(ACFMAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=(3,3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1,1,1), bias=bias)
        self.fc = nn.Conv3d(3*self.num_heads, 9, kernel_size=(1,1,1), bias=True)

        self.dep_conv = nn.Conv3d(9*dim//self.num_heads, dim, kernel_size=(3,3,3), bias=True, groups=dim//self.num_heads, padding=1)


    def forward(self, x):
        b,c,h,w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0,2,3,1)
        f_all = qkv.reshape(f_conv.shape[0], h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        #local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[1]//self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv) # B, C, H, W
        out_conv = out_conv.squeeze(2)


        # global SA
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output =  out + out_conv

        return output


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# A dual encoder crack segmentation network with Haar wavelet-based high–low frequency attention
class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out


class FFM(nn.Module):
    def __init__(self, dim1):
        super().__init__()
        self.trans_c = nn.Conv2d(dim1, dim1, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim1, dim1)
        self.li2 = nn.Linear(dim1, dim1)

        self.qx = DSC(dim1, dim1)
        self.kx = DSC(dim1, dim1)
        self.vx = DSC(dim1, dim1)
        self.projx = DSC(dim1, dim1)

        self.qy = DSC(dim1, dim1)
        self.ky = DSC(dim1, dim1)
        self.vy = DSC(dim1, dim1)
        self.projy = DSC(dim1, dim1)

        self.concat = nn.Conv2d(dim1 * 2, dim1, 1)

        self.fusion = nn.Sequential(IDSC(dim1 * 4, dim1),
                                    nn.BatchNorm2d(dim1),
                                    nn.GELU(),
                                    DSC(dim1, dim1),
                                    nn.BatchNorm2d(dim1),
                                    nn.GELU(),
                                    nn.Conv2d(dim1, dim1, 1),
                                    nn.BatchNorm2d(dim1),
                                    nn.GELU())

    def forward(self, x1):
        x, y = x1
        y = y.reshape(y.size()[0], -1, y.size()[1])
        b, c, h, w = x.shape
        B, N, C = y.shape
        H = W = int(N ** 0.5)

        x = self.trans_c(x)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)

        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight

        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        qy = self.qy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        kx = self.kx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vx = self.vx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attnx = (qy @ kx.transpose(-2, -1)) * (C ** -0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)

        qx = self.qx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        ky = self.ky(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vy = self.vy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attny = (qx @ ky.transpose(-2, -1)) * (C ** -0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)

        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)

        out = torch.cat([x, y, out1, out2], dim=1)

        out = self.fusion(out)
        return out


# https://arxiv.org/abs/2501.10040
# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
class PA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            norm_layer(dim * 4),
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)

        return x


class LA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MRA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))

        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)

        out = x[:, :self.channel, :, :] * F.interpolate(self.gate_fn(att),
                                                        size=(x.shape[-2], x.shape[-1]),
                                                        mode='nearest')
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class GA12(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)
        x = self.uppool(x_, indices=idx)
        return x


class D_GA(nn.Module):

    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = GA(dim)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x = self.norm(self.attn(x_))
        x = self.uppool(x, indices=idx)

        return x


class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class LWGA_Block(nn.Module):
    def __init__(self,
                 dim,
                 stage=2,
                 att_kernel=3,
                 mlp_ratio=4.0,
                 drop_path=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d
                 ):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.PA = PA(self.dim_split, norm_layer, act_layer)  # PA is point attention
        self.LA = LA(self.dim_split, norm_layer, act_layer)  # LA is local attention
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)  # MRA is medium-range attention
        if stage == 2:
            self.GA3 = D_GA(self.dim_split, norm_layer)  # GA3 is global attention (stage of 3)
        elif stage == 3:
            self.GA4 = GA(self.dim_split)  # GA4 is global attention (stage of 4)
            self.norm = norm_layer(self.dim_split)
        else:
            self.GA12 = GA12(self.dim_split, act_layer)  # GA12 is global attention (stages of 1 and 2)
            self.norm = norm_layer(self.dim_split)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # for training/inference
        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_split, self.dim_split, self.dim_split, self.dim_split], dim=1)
        x1 = x1 + self.PA(x1)
        x2 = self.LA(x2)
        x3 = self.MRA(x3)
        if self.stage == 2:
            x4 = x4 + self.GA3(x4)
        elif self.stage == 3:
            x4 = self.norm(x4 + self.GA4(x4))
        else:
            x4 = self.norm(x4 + self.GA12(x4))
        x_att = torch.cat((x1, x2, x3, x4), 1)

        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))

        return x


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343\
# SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution
# https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
# https://github.com/Zheng-MJ/SMFANet
class DMlp_SMFA(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class SMFA(nn.Module):
    def __init__(self, dim):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.lde = DMlp_SMFA(dim, 2)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()
        self.down_scale = 8
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))  # 乘法因子
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))  # 加法因子

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w), mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343\
# Dual selective fusion transformer network for hyperspectral image classification
# https://www.sciencedirect.com/science/article/abs/pii/S089360802500190X
# https://github.com/YichuXu/DSFormer
class KernelSelectiveFusionAttention(nn.Module):
    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        dim = x.size(1)
        attn1 = self.conv0(x)  # conv_3*3
        attn2 = self.conv_spatial(attn1)  # conv_3*3 -> conv_5*5

        attn1 = self.conv1(attn1) # b, dim/2, h, w
        attn2 = self.conv2(attn2) # b, dim/2, h, w

        attn = torch.cat([attn1, attn2], dim=1)  # b,c,h,w
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # b,1,h,w
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # b,1,h,w
        agg = torch.cat([avg_attn, max_attn], dim=1) # spa b,2,h,w

        ch_attn1 = self.global_pool(attn) # b,dim,1, 1
        z = self.fc1(ch_attn1)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)
        a_b = self.softmax(a_b)

        a1,a2 =  a_b.chunk(2, dim=1)
        a1 = a1.reshape(batch_size,dim // 2,1,1)
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)

        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1)

        attn = attn1 * w1 + attn2 * w2
        attn = self.conv(attn).sigmoid()

        return x * attn


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343\
# Token Statistics Transformer Linear-Time Attention via Variational Rate Reduction
# https://arxiv.org/abs/2412.17810
# https://robinwu218.github.io/ToST/
class Token_Selective_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False, k=0.8, group_num=4):
        super(Token_Selective_Attention, self).__init__()
        self.num_heads = num_heads
        self.k = k
        self.group_num = group_num
        self.dim_group = dim // group_num
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv3d(self.group_num, self.group_num * 3, kernel_size=(1, 1, 1), bias=False)
        self.qkv_conv = nn.Conv3d(self.group_num * 3, self.group_num * 3, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                  groups=self.group_num * 3, bias=bias)  # 331
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b,self.group_num,c//self.group_num,h,w)
        b, t, c, h, w = x.shape  # 2,4,32,8,8

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = rearrange(q, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        k = rearrange(k, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)
        v = rearrange(v, 'b t (head c) h w -> b head c (h w t)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, _, N = q.shape  # N=hw

        mask = torch.zeros(b, self.num_heads, N, N, device=x.device, requires_grad=False)

        attn = (q.transpose(-2, -1) @ k) * self.temperature  # [b, hw, hw]

        index = torch.topk(attn, k=int(N * self.k), dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # [b, c, hw]

        out = rearrange(out, 'b head c (h w t) -> b t (head c) h w', head=self.num_heads, h=h, w=w)

        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)

        return out


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343\
# ASCNet_Asymmetric_Sampling_Correction_Network_for_Infrared_Image_Destriping
# https://ieeexplore.ieee.org/document/10855453
# https://github.com/xdFai/ASCNet/tree/main
class ChannelPool(nn.Module):
    def forward(self, x):
        # 将maxpooling 与 global average pooling 结果拼接在一起
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))

        self.conv_1x1 = nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=1, padding=0, stride=1,
                                  bias=False)
        self.bn = nn.BatchNorm2d(2 * channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.LeakyReLU()

        self.F_h = nn.Sequential(  # 激发操作
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.F_w = nn.Sequential(  # 激发操作
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()
        res = x
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))
        x_1, x_2 = x.split(C, 1)

        x_1 = self.F_h(x_1)
        x_2 = self.F_w(x_2)
        s_h = self.sigmoid(x_1)
        s_w = self.sigmoid(x_2)

        out = res * s_h.expand_as(res) * s_w.expand_as(res)

        return out


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=False, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class RCSSC(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(RCSSC, self).__init__()
        pooling_r = 4
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),
        )
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(n_feat)
        )
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1),
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU()
        self.tail = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = self.head(x)
        sa_branch = self.SA(x)
        ca_branch = self.CA(x)
        x1 = torch.cat([sa_branch, ca_branch], dim=1)  # 拼接
        x1 = self.conv1x1(x1)
        x2 = torch.sigmoid(
            torch.add(x, F.interpolate(self.SC(x), x.size()[2:])))
        out = torch.mul(x1, x2)
        out = self.tail(out)
        out = out + res
        out = self.ReLU(out)
        return out


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343\
#  (TGRS 2025) An Adaptive Dual-Supervised Cross-Deep Dependency Network for Pixel-Wise Classification
# https://github.com/ChenC1027/ADCD-Net/tree/main
class DeformableInteractiveAttention(nn.Module):
    def __init__(self, stride=1, distortionmode=False):
        super(DeformableInteractiveAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.distortionmode = distortionmode
        self.upsample = nn.Upsample(scale_factor=2)
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        if distortionmode:
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)
            self.d_conv.register_full_backward_hook(self._set_lra)

            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrm)

    @staticmethod
    def _set_lra(module, grad_input, grad_output):
        grad_input = [g * 0.4 if g is not None else None for g in grad_input]
        grad_output = [g * 0.4 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input

    @staticmethod
    def _set_lrm(module, grad_input, grad_output):
        grad_input = [g * 0.1 if g is not None else None for g in grad_input]
        grad_output = [g * 0.1 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        avg_out = self.downavg(avg_out)
        max_out = self.downmax(max_out)

        out = torch.cat([max_out, avg_out], dim=1)

        if self.distortionmode:
            d_avg_out = torch.sigmoid(self.d_conv(avg_out))
            d_max_out = torch.sigmoid(self.d_conv1(max_out))
            out = torch.cat([d_avg_out * max_out, d_max_out * avg_out], dim=1)

        out = self.conv(out)
        mask = self.sigmoid(self.upsample(out))
        att_out = x * mask
        return F.relu(att_out)

# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
class ContMix(nn.Module):
    def __init__(self,
                 dim=64,
                 ctx_dim=32,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 ):
        super().__init__()
        ctx_dim = dim // 2
        self.kernel_size = kernel_size
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.weight_query = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=False),  # 32 -> 32
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_proj = nn.Conv2d(49, kernel_size ** 2 + smk_size ** 2, kernel_size=1)

        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.get_rpb()

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size // 2] = height - (kernel_size - 1)
        num_repeat_w[kernel_size // 2] = width - (kernel_size - 1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (
                    2 * kernel_size - 1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size ** 2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size ** 2))
        return attn + rpb

    def forward(self, x):
        B, C, H, W = x.shape

        query, key = torch.chunk(x, 2, dim=1)  # 32, 32
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)
        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)

        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size ** 2, self.kernel_size ** 2], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)

        x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
        x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)

        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)
        x = self.dyconv_proj(x)
        return x


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343\
# (TGRS 2025) CFFormer_A_Cross-Fusion_Transformer_Framework_for_the_Semantic_Segmentation_of_Multisource_Remote_Sensing_Images
# https://ieeexplore.ieee.org/document/10786275
# https://github.com/masurq/CFFormer
import math
from timm.models.layers import trunc_normal_

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self,x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)  # B 6C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


# 先空间校正再通道校正
class FCM(nn.Module):
    def __init__(self, dim, reduction=1, eps=1e-8):
        super(FCM, self).__init__()
        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x1, x2 = x
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        spatial_weights = self.spatial_weights(x1, x2)
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1

        channel_weights = self.channel_weights(x1_1, x2_1)

        main_out = x1_1 + fuse_weights[1] * channel_weights[1] * x2_1
        aux_out = x2_1 + fuse_weights[1] * channel_weights[0] * x1_1
        # return main_out, aux_out
        return torch.cat([main_out, aux_out], dim=1)


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343\
# https://github.com/ooo1128/BAFNet
# (TGRS 2025) Boundary-Aware Feature Fusion With Dual-Stream Attention for Remote Sensing Small Object Detectio
# https://ieeexplore.ieee.org/document/10787034
class Pred_Layer(nn.Module):
    def __init__(self, in_c=256):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x1 = self.outlayer(x)
        return x, x1


class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


# Dual-Stream Attention Module
class DSAM(nn.Module):
    def __init__(self, in_c):
        super(DSAM, self).__init__()
        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.rgbd_pred_layer = Pred_Layer(256 * 8)
        self.conv = nn.Conv2d(in_c, 1, 1, 1, 0,)

    def forward(self, x):
        pred = self.conv(x)
        feat = x
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        enhanced_feat, new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        # return enhanced_feat, new_pred
        return enhanced_feat


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343\
# EGNet Lightweight Edge-Gaussian Driven Network for Low-Quality Remote Sensing Image Object Detection
# https://github.com/lwCVer/LEGNet
# https://arxiv.org/abs/2503.14012
class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])

    def forward(self, x):
        out = self.block(x)
        return out


class EdgeGaussianAggregation(nn.Module):
    def __init__(self, dim, size=3, sigma=1, norm_layer=dict(type='BN', requires_grad=True), act_layer=nn.ReLU,
                 feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# SalM²: An Extremely Lightweight Saliency Mamba Model for Real-Time Cognitive Awareness of Driver Attention
# https://github.com/zhao-chunyu/SaliencyMamba
# https://ojs.aaai.org/index.php/AAAI/article/view/32157
class CrossModelAtt(nn.Module):
    def __init__(self,):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        img_feat, text_feat = x
        B, C, H, W = img_feat.shape
        q = img_feat.view(B, C, -1)
        k = text_feat.view(B, C, -1).permute(0, 2, 1)
        attention_map = torch.bmm(q, k)  # [B, C, C]
        attention_map = self.softmax(attention_map)
        v = text_feat.view(B, C, -1)
        attention_info = torch.bmm(attention_map, v)
        attention_info = attention_info.view(B, C, H, W)
        output = self.gamma * attention_info + img_feat

        return output


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# A Hybrid Transformer-Mamba Network for Single Image Deraining
# https://github.com/sunshangquan/TransMamba
# https://arxiv.org/abs/2409.00410
class SBSAtt(nn.Module):
    def __init__(self, dim, num_heads=2, bias=True):
        super(SBSAtt, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.factor = 2
        self.idx_dict = {}
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[..., t_pad[0]:hw - t_pad[1]]

    def comp2real(self, x):
        b, _, h, w = x.shape
        return torch.cat([x.real, x.imag], 1)

    #        return torch.stack([x.real, x.imag], 2).view(b,-1,h,w)
    def real2comp(self, x):
        xr, xi = x.chunk(2, dim=1)
        return torch.complex(xr, xi)

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def get_idx_map(self, h, w):
        l1_u = torch.arange(h // 2).view(1, 1, -1, 1)
        l2_u = torch.arange(w).view(1, 1, 1, -1)
        half_map_u = l1_u @ l2_u
        l1_d = torch.arange(h - h // 2).flip(0).view(1, 1, -1, 1)
        l2_d = torch.arange(w).view(1, 1, 1, -1)
        half_map_d = l1_d @ l2_d
        return torch.cat([half_map_u, half_map_d], 2).view(1, 1, -1).argsort(-1)

    def get_idx(self, x):
        h, w = x.shape[-2:]
        if (h, w) in self.idx_dict:
            return self.idx_dict[(h, w)]
        idx_map = self.get_idx_map(h, w).to(x.device).detach()
        self.idx_dict[(h, w)] = idx_map
        return idx_map

    def attn(self, qkv):
        h = qkv.shape[2]
        q, k, v = qkv.chunk(3, dim=1)

        q, pad_w, idx = self.fft(q)
        q, pad = self.pad(q, self.factor)
        k, pad_w, _ = self.fft(k)
        k, pad = self.pad(k, self.factor)
        v, pad_w, _ = self.fft(v)
        v, pad = self.pad(v, self.factor)

        q = rearrange(q, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        k = rearrange(k, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        v = rearrange(v, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (c factor) hw -> b (head c) (factor hw)', head=self.num_heads, factor=self.factor)
        out = self.unpad(out, pad)
        out = self.ifft(out, pad_w, idx, h)
        return out

    def fft(self, x):
        x, pad = self.pad(x, 2)
        x = torch.fft.rfft2(x.float(), norm="ortho")
        x = self.comp2real(x)
        idx = self.get_idx(x)
        b, c = x.shape[:2]
        x = x.contiguous().view(b, c, -1)
        x = torch.gather(x, 2, index=idx.repeat(b, c, 1))  # b, 6c, h*(w//2+1)
        return x, pad, idx

    def ifft(self, x, pad, idx, h):
        b, c = x.shape[:2]
        x = torch.scatter(x, 2, idx.repeat(b, c, 1), x)
        x = x.view(b, c, h, -1)
        x = self.real2comp(x)
        x = torch.fft.irfft2(x, norm='ortho')  # .abs()
        x = self.unpad(x, pad)
        return x

    def forward(self, x):
        qkv = self.qkv_dwconv(self.qkv(x))

        out = self.attn(qkv)
        out = self.project_out(out)
        return out


# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# https://arxiv.org/pdf/2503.11030?
# FMNet: Frequency-Assisted Mamba-Like Linear Attention Network for Camouflaged Object Detection
from einops import rearrange, repeat
from torch.nn import Softmax

def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor

class FrequencyAttention(nn.Module):
    def __init__(self, in_dim):
        super(FrequencyAttention, self).__init__()

        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=3, padding=3), nn.BatchNorm2d(down_dim), nn.ReLU(True)
        )
        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.temperature = nn.Parameter(torch.ones(8, 1, 1))

        self.weight = nn.Sequential(
            nn.Conv2d(down_dim, down_dim // 16, 1, bias=True),
            nn.BatchNorm2d(down_dim // 16),
            nn.ReLU(True),
            nn.Conv2d(down_dim // 16, down_dim, 1, bias=True),
            nn.Sigmoid())

        self.softmax = Softmax(dim=-1)
        self.norm = nn.BatchNorm2d(down_dim)
        self.relu = nn.ReLU(True)
        self.num_heads = 8

    def forward(self, x):

        conv2 = self.conv2(x)
        b, c, h, w = conv2.shape

        q_f_2 = torch.fft.fft2(conv2.float())
        k_f_2 = torch.fft.fft2(conv2.float())
        v_f_2 = torch.fft.fft2(conv2.float())
        tepqkv = torch.fft.fft2(conv2.float())

        q_f_2 = rearrange(q_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f_2 = rearrange(k_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f_2 = rearrange(v_f_2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f_2 = torch.nn.functional.normalize(q_f_2, dim=-1)
        k_f_2 = torch.nn.functional.normalize(k_f_2, dim=-1)
        attn_f_2 = (q_f_2 @ k_f_2.transpose(-2, -1)) * self.temperature
        attn_f_2 = custom_complex_normalization(attn_f_2, dim=-1)
        out_f_2 = torch.abs(torch.fft.ifft2(attn_f_2 @ v_f_2))
        out_f_2 = rearrange(out_f_2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l_2 = torch.abs(torch.fft.ifft2(self.weight(tepqkv.real)*tepqkv))
        out_2 = torch.cat((out_f_2,out_f_l_2),1)

        F_2 = torch.add(out_2, x)

        return F_2


# MogaNet: Multi-order Gated Aggregation Network
# https://arxiv.org/pdf/2211.03295
# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
# https://github.com/AIFengheshu/Plug-play-modules
def build_act_layer(act_type):
    #Build activation layer
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
    #A learnable element-wise scaler.

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = int(embed_dims * 4)

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

# https://blog.csdn.net/StopAndGoyyy?spm=1011.2124.3001.5343
__all__ = ['SCBottleneck', 'align_3In', 'RepVGGBlocks', 'SPPF_WD', 'RCRep2A', 'C2f_AkConv', 'C3_AKConv',
           'RCRep2A_AKConv', 'C3k2_AKConv', 'MetaNeXt', 'ConvNeXt', 'InceptionNeXtBlock', 'InceptionDWConv2d',
           'C2f_InceptionDWConv2d', 'C3_InceptionDWConv2d', 'C3k2_InceptionDWConv2d', 'RCRep2A_InceptionDWConv2d',
           'FRFN', 'C2f_FRFN', 'C3_FRFN', 'C3k2_FRFN', 'RCRep2A_FRFN', 'SMFA', 'RCRep2A_SMFA', 'C2f_SMFA', 'C3_SMFA',
           'C3k2_SMFA', 'C3_ScConv', 'C3k2_ScConv', 'C2f_ScConv', 'RCRep2A_DySnakeConv', 'C3k2_DySnakeConv',
           'C3_DySnakeConv', 'C2f_DySnakeConv', 'RCRep2A_SCConv', 'C2f_SCConv', 'C3_SCConv', 'C3k2_SCConv',
           'C3k2_ScConv', 'C3_ScConv', 'C2f_ScConv', 'RCRep2A_ScConv', 'iRMB', 'RCRep2A_iRMB', 'C2f_iRMB', 'C3_iRMB',
           'C3k2_iRMB', 'RCRep2A_PKIBlock', 'C2f_PKIBlock', 'C3_PKIBlock', 'C3k2_PKIBlock',
           # 'AdaptiveDilatedConv', 'PKIBlock', 'Mona'
           'GhostModuleV3', 'PConv', 'SCAM', 'FEM', 'FFM_Concat2', 'FFM_Concat3', 'RCRep2A_DEABlock', 'C2f_DEABlock',
           'C3_DEABlock', 'C3k2_DEABlock', 'DEABlock', 'DEBlock', 'SKFusion', 'gnconv', 'RCRep2A_gnconv', 'C2f_gnconv',
           'C3_gnconv', 'C3k2_gnconv', 'RCRep2A_RCAB', 'C2f_RCAB', 'C3_RCAB', 'C3k2_RCAB', 'RCAB', 'RCRep2A_RCBv6',
           'C2f_RCBv6', 'C3_RCBv6', 'C3k2_RCBv6', 'RCBv6', 'RCRep2A_MLKA_Ablation', 'C2f_MLKA_Ablation',
           'C3_MLKA_Ablation', 'C3k2_MLKA_Ablation', 'MLKA_Ablation', 'RCRep2A_ConvMod', 'C2f_ConvMod', 'C3_ConvMod',
           'C3k2_ConvMod', 'ConvMod', 'PagFM', 'WCMF', 'RCRep2A_WTConv2d', 'C2f_WTConv2d', 'C3_WTConv2d',
           'C3k2_WTConv2d', 'RCRep2A_ConvolutionalGLU', 'C2f_ConvolutionalGLU', 'C3_ConvolutionalGLU',
           'ConvolutionalGLU', 'C3k2_ConvolutionalGLU', 'RCRep2A_RCM', 'C2f_RCM', 'C3_RCM', 'C3k2_RCM', 'RCM', 'EUCB',
           'LGAG', 'MSCB', 'MSDC', 'star_Block', 'CARAFE', 'DySample', 'ds_', 'RCRep2A_FFCM', 'C2f_FFCM', 'C3_FFCM',
           'C3k2_FFCM', 'FFCM', 'RCRep2A_FSAS', 'C2f_FSAS', 'C3_FSAS', 'C3k2_FSAS', 'FSAS', 'SDI', 'Ghost_SDI',
           'SimFusion_4in', 'AdvPoolFusion', 'SimFusion_3in', 'IFM', 'InjectionMultiSum_Auto_pool', 'PyramidPoolAgg',
           'TopBasicLayer', 'RCRep2A_LSK', 'C2f_LSK', 'C3_LSK', 'C3k2_LSK', 'RCRep2A_SFH_former_Block',
           'C2f_SFH_former_Block', 'C3_SFH_former_Block', 'C3k2_SFH_former_Block', 'SFH_former_Block', 'BiFPN', 'AFPN',
           'get_feturemap', 'RCRep2A_MDHTA', 'C2f_MDHTA', 'C3_MDHTA', 'C3k2_MDHTA', 'MDHTA', 'ADown_light',
           'ConDSeg_model', 'CDFAPreprocess', 'global_meta_block', 'DilatedMDTA', 'RCRep2A_DilatedMDTA',
           'C2f_DilatedMDTA', 'C3_DilatedMDTA', 'C3k2_DilatedMDTA', 'LHConcat', 'LowDctFrequencyExtractor',
           'HighDctFrequencyExtractor', 'SABlock', 'RCRep2A_A2', 'C3_R_ELAN', 'GBC', 'RCRep2A_GBC', 'C2f_GBC', 'C3_GBC',
           'C3k2_GBC', 'ARConv', 'Conv_DyT', 'I_LCA', 'HV_LCA', 'RCRep2A_HV_LCA', 'C2f_HV_LCA', 'C3_HV_LCA',
           'C3k2_HV_LCA', 'RCRep2A_MuLUTUnit', 'C2f_MuLUTUnit', 'C3_MuLUTUnit', 'C3k2_MuLUTUnit', 'EBlock', 'DBlock',
           'token_mixer', 'RCRep2A_token_mixer', 'C2f_token_mixer', 'C3_token_mixer', 'C3k2_token_mixer', 'PConv2',
           'RepConvBlock', 'ResDWConv', 'DilatedReparamBlock', 'GRN', 'SEModule', 'RCRep2A_MuLUTUnit', 'C2f_MuLUTUnit',
           'C3_MuLUTUnit', 'C3k2_MuLUTUnit', 'MuLUTUnit', 'Octave', 'HPA', 'TBFE', 'EfficientViMBlock', 'HRAMi',
           'Downsizing', 'SHSA', 'FeedForward_SEF', 'FeatureCorrection_s2c', 'SequenceShuffleAttention', 'EDFFN',
           'ACFMAttention', 'FFM', 'LWGA_Block', 'SMFA', 'KernelSelectiveFusionAttention', 'Token_Selective_Attention',
           'RCSSC', 'DeformableInteractiveAttention', 'ContMix', 'FCM', 'DSAM', 'EdgeGaussianAggregation',
           'CrossModelAtt', 'SBSAtt', 'FrequencyAttention', 'ChannelAggregationFFN'
           ]


if __name__ == '__main__':
    x = torch.randn(1, 256, 16, 16)
    model = RCRep2A(256, 512)
    print(model(x,).shape)

