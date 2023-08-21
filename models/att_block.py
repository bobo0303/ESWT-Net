import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math
from einops import rearrange


######################################################################################
# Attention-Aware Layer
######################################################################################
class AttnAware(nn.Module):
    def __init__(self, input_nc, activation='gelu', norm='pixel', num_heads=2):
        super(AttnAware, self).__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)
        head_dim = input_nc // num_heads
        self.num_heads = num_heads
        self.input_nc = input_nc
        self.scale = head_dim ** -0.5

        self.query_conv = nn.Sequential(
            norm_layer(input_nc),
            activation_layer,
            nn.Conv2d(input_nc, input_nc, kernel_size=1)
        )
        self.key_conv = nn.Sequential(
            norm_layer(input_nc),
            activation_layer,
            nn.Conv2d(input_nc, input_nc, kernel_size=1)
        )

        self.weight = nn.Conv2d(self.num_heads*2, 2, kernel_size=1, stride=1)
        self.to_out = ResnetBlock(input_nc * 2, input_nc, 1, 0, activation, norm)

    def forward(self, x, att=None, pre=None, mask=None):
        B, C, W, H = x.size()
        q = self.query_conv(x).view(B, -1, W*H)
        k = self.key_conv(x).view(B, -1, W*H)
        v = x.view(B, -1, W*H)

        q = rearrange(q, 'b (h d) n -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b (h d) n -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b (h d) n -> b h n d', h=self.num_heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        if att != None:
            attn = attn + att[0]
            attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij, bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n').view(B, -1, W, H)

        out = self.to_out(torch.cat([out, x], dim=1))
        return out

######################################################################################
# base modules
######################################################################################

class ResnetBlock(nn.Module):
    def __init__(self, input_nc, output_nc=None, kernel=3, dropout=0.0, activation='gelu', norm='pixel', return_mask=False):
        super(ResnetBlock, self).__init__()

        activation_layer = get_nonlinearity_layer(activation)
        norm_layer = get_norm_layer(norm)
        self.return_mask = return_mask

        output_nc = input_nc if output_nc is None else output_nc

        self.norm1 = norm_layer(input_nc)
        self.conv1 = PartialConv2d(input_nc, output_nc, kernel_size=kernel, padding=int((kernel-1)/2), return_mask=True)
        self.norm2 = norm_layer(output_nc)
        self.conv2 = PartialConv2d(output_nc, output_nc, kernel_size=kernel, padding=int((kernel-1)/2), return_mask=True)
        self.dropout = nn.Dropout(dropout)
        self.act = activation_layer

        if input_nc != output_nc:
            self.short = PartialConv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0)
        else:
            self.short = Identity()

    def forward(self, x, mask=None):
        x_short = self.short(x)
        x, mask = self.conv1(self.act(self.norm1(x)), mask)
        x, mask = self.conv2(self.dropout(self.act(self.norm2(x))), mask)
        if self.return_mask:
            return (x + x_short) / math.sqrt(2), mask
        else:
            return (x + x_short) / math.sqrt(2)

######################################################################################
# base function for network structure
######################################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'pixel':
        norm_layer = functools.partial(PixelwiseNorm)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'relu':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'gelu':
        nonlinearity_layer = nn.GELU()
    elif activation_type == 'leakyrelu':
        nonlinearity_layer = nn.LeakyReLU(0.2)
    elif activation_type == 'prelu':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


class PixelwiseNorm(nn.Module):
    def __init__(self, input_nc):
        super(PixelwiseNorm, self).__init__()
        self.init = False
        self.alpha = nn.Parameter(torch.ones(1, input_nc, 1, 1))

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        # x = x - x.mean(dim=1, keepdim=True)
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).rsqrt()  # [N1HW]
        y = x * y  # normalize the input x volume
        return self.alpha*y


###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask1 = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask1)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask1)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask / self.slide_winsize   # replace the valid value to confident score
        else:
            return output