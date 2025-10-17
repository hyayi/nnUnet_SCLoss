# complete_dsc_residual_unet.py
"""
Complete DSC (Dynamic Snake Convolution) Enhanced Residual UNet
nnU-Net compatible implementation for tubular structure segmentation

Author: Based on nnU-Net framework with DSConv integration
Date: 2025-10-17
"""

from typing import Union, Type, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

# Assume DSConv is imported from the original implementation
# from S3_DSConv import DSConv

# ========================================
# DSConv Class (included for completeness)
# ========================================

import warnings
warnings.filterwarnings("ignore")

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph,
                 if_offset, device):
        """
        The Dynamic Snake Convolution
        :param in_ch: input channel
        :param out_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param morph: the morphology of the convolution kernel is mainly divided into two types
                        along the x-axis (0) and the y-axis (1) (see the paper for details)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        """
        super(DSConv, self).__init__()
        # use the <offset_conv> to learn the deformable offset
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two types of the DSConv (along x-axis and y-axis)
        self.dsc_conv_x = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        # We need a range of deformation between -1 and 1 to mimic the snake's swing
        offset = torch.tanh(offset)
        input_shape = f.shape
        dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph,
                  self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dsc_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dsc_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x

# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope

        # define feature map shape
        """
        B: Batch size  C: Channel  W: Width  H: Height
        """
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            """
            Initialize the kernel and flatten the kernel
                y: only need 0
                x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                !!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
            """
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1).to(self.device)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1).to(self.device)

            y_offset_new = y_offset.detach().clone()

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)

                # The center position remains unchanged and the rest of the positions begin to swing
                # This part is quite simple. The main idea is that "offset is an iterative process"
                y_offset_new[center] = 0
                for index in range(1, center):
                    y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
                    y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(self.device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, self.num_points * self.width, 1 * self.height
            ])
            return y_new, x_new

        else:
            """
            Initialize the kernel and flatten the kernel
                y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
                x: only need 0
            """
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)

            y, x = torch.meshgrid(y, x)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset_new = x_offset.detach().clone()

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset_new.permute(1, 0, 2, 3)
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center):
                    x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
                    x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(self.device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            y_new = y_new.permute(0, 3, 1, 4, 2)
            y_new = y_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height])
            x_new = x_new.permute(0, 3, 1, 4, 2)
            x_new = x_new.reshape([
                self.num_batch, 1 * self.width, self.num_points * self.height
            ])
            return y_new, x_new

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = torch.ones([self.num_points * self.width * self.height
                             ]).unsqueeze(0)
        repeat = repeat.float()

        base = torch.matmul(base, repeat)
        base = base.reshape([-1])

        base = base.to(self.device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 - base + x0
        index_c0 = base_y0 - base + x1

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 - base + x0
        index_c1 = base_y1 - base + x1

        # get 8 grid values
        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(self.device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(self.device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(self.device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(self.device)

        # find 8 grid locations
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(self.device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(self.device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(self.device)

        outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
                   value_c1 * vol_c1)

        if self.morph == 0:
            outputs = outputs.reshape([
                self.num_batch,
                self.num_points * self.width,
                1 * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape([
                self.num_batch,
                1 * self.width,
                self.num_points * self.height,
                self.num_channels,
            ])
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, y, x)
        return deformed_feature

# ========================================
# 1. DSC Building Blocks
# ========================================

class DSCBlock(nn.Module):
    """
    DSC block that combines x-axis and y-axis DSConv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 7,
                 extend_scope: float = 1.0,
                 if_offset: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        self.dsconv_x = DSConv(in_channels, out_channels, kernel_size, extend_scope, 0, if_offset, device)
        self.dsconv_y = DSConv(in_channels, out_channels, kernel_size, extend_scope, 1, if_offset, device)
        
        # Feature fusion
        self.fusion = nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0)
        
    def forward(self, x):
        x_dsc_x = self.dsconv_x(x)
        x_dsc_y = self.dsconv_y(x)
        
        # Size matching if needed
        if x_dsc_x.shape != x_dsc_y.shape:
            target_size = min(x_dsc_x.shape[2], x_dsc_y.shape[2])
            x_dsc_x = F.adaptive_avg_pool2d(x_dsc_x, target_size)
            x_dsc_y = F.adaptive_avg_pool2d(x_dsc_y, target_size)
        
        x_fused = torch.cat([x_dsc_x, x_dsc_y], dim=1)
        return self.fusion(x_fused)

class StandardConvBlock(nn.Module):
    """Standard convolution block"""
    def __init__(self, in_ch, out_ch, kernel_size, conv_op, conv_bias, norm_op, norm_op_kwargs, 
                 dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first):
        super().__init__()
        
        self.nonlin_first = nonlin_first
        
        self.conv = conv_op(in_ch, out_ch, kernel_size, 1, kernel_size//2, bias=conv_bias)
        
        if norm_op is not None:
            self.norm = norm_op(out_ch, **norm_op_kwargs)
        else:
            self.norm = None
            
        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
        else:
            self.nonlin = None
            
        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
    
    def forward(self, x):
        if self.nonlin_first:
            if self.nonlin is not None:
                x = self.nonlin(x)
            if self.norm is not None:
                x = self.norm(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            if self.norm is not None:
                x = self.norm(x)
            if self.nonlin is not None:
                x = self.nonlin(x)
                
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x

class DSCEnhancedConv(nn.Module):
    """DSC enhanced convolution block"""
    def __init__(self, in_ch, out_ch, kernel_size, conv_bias, norm_op, norm_op_kwargs, 
                 dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                 extend_scope, if_offset, device):
        super().__init__()
        
        self.nonlin_first = nonlin_first
        
        # Standard conv
        self.std_conv = nn.Conv2d(in_ch, out_ch, kernel_size, 1, kernel_size//2, bias=conv_bias)
        
        # DSC enhancement
        self.dsc_block = DSCBlock(in_ch, out_ch, kernel_size=7, extend_scope=extend_scope, 
                                if_offset=if_offset, device=device)
        
        # Feature fusion
        self.fusion = nn.Conv2d(2 * out_ch, out_ch, 1, 1, 0, bias=conv_bias)
        
        if norm_op is not None:
            self.norm = norm_op(out_ch, **norm_op_kwargs)
        else:
            self.norm = None
            
        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
        else:
            self.nonlin = None
            
        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
    
    def forward(self, x):
        # Standard conv path
        std_out = self.std_conv(x)
        
        # DSC enhanced path
        dsc_out = self.dsc_block(x)
        
        # Size matching
        if dsc_out.shape[2:] != std_out.shape[2:]:
            dsc_out = F.adaptive_avg_pool2d(dsc_out, std_out.shape[2:])
        
        # Feature fusion
        fused = torch.cat([std_out, dsc_out], dim=1)
        out = self.fusion(fused)
        
        # Apply norm/nonlin
        if self.nonlin_first:
            if self.nonlin is not None:
                out = self.nonlin(out)
            if self.norm is not None:
                out = self.norm(out)
        else:
            if self.norm is not None:
                out = self.norm(out)
            if self.nonlin is not None:
                out = self.nonlin(out)
                
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out

# ========================================
# 2. DSC Residual Blocks
# ========================================

class HybridDSCBasicBlockD(nn.Module):
    """
    Hybrid DSC Residual Block
    """
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 conv_op: Type[_ConvNd],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 extend_scope: float = 1.0,
                 if_offset: bool = True,
                 device: str = 'cuda',
                 use_dsc: bool = True):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}

        self.stride = stride
        self.use_dsc = use_dsc and (conv_op == nn.Conv2d)  # Only use DSC for 2D

        # First conv - standard
        self.conv1 = conv_op(in_planes, planes, 3, stride, 1, bias=conv_bias)
        if norm_op is not None:
            self.bn1 = norm_op(planes, **norm_op_kwargs)
        else:
            self.bn1 = None
        if nonlin is not None:
            self.relu1 = nonlin(**nonlin_kwargs)
        else:
            self.relu1 = None

        # Second conv - DSC enhanced or standard
        if self.use_dsc:
            # Use DSC block for tubular structure enhancement
            self.dsc_block = DSCBlock(planes, planes, kernel_size=7, extend_scope=extend_scope, 
                                    if_offset=if_offset, device=device)
            self.conv2 = None
        else:
            # Standard conv
            self.conv2 = conv_op(planes, planes, 3, 1, 1, bias=conv_bias)
            self.dsc_block = None
            
        if norm_op is not None:
            self.bn2 = norm_op(planes, **norm_op_kwargs)
        else:
            self.bn2 = None

        # Dropout
        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None

        # Shortcut connection
        if in_planes != planes or stride != 1:
            self.downsample = conv_op(in_planes, planes, 1, stride, 0, bias=conv_bias)
            if norm_op is not None:
                self.downsample_norm = norm_op(planes, **norm_op_kwargs)
            else:
                self.downsample_norm = None
        else:
            self.downsample = None
            self.downsample_norm = None

        # Final activation
        if nonlin is not None:
            self.final_relu = nonlin(**nonlin_kwargs)
        else:
            self.final_relu = None

    def forward(self, x):
        residual = x
        
        # First conv
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        if self.relu1 is not None:
            out = self.relu1(out)
            
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
            
        # Second conv - DSC enhanced or standard
        if self.use_dsc:
            # DSC block with size matching
            dsc_out = self.dsc_block(out)
            # Match size to standard conv output
            if dsc_out.shape[2:] != out.shape[2:]:
                dsc_out = F.adaptive_avg_pool2d(dsc_out, out.shape[2:])
            out = dsc_out
        else:
            out = self.conv2(out)
            
        if self.bn2 is not None:
            out = self.bn2(out)
        
        # Shortcut connection
        if self.downsample is not None:
            residual = self.downsample(x)
            if self.downsample_norm is not None:
                residual = self.downsample_norm(residual)
        
        # Add residual
        out += residual
        
        # Final activation
        if self.final_relu is not None:
            out = self.final_relu(out)
        
        return out

class DSCStackedResidualBlocks(nn.Module):
    """
    DSC version of StackedResidualBlocks
    """
    def __init__(self,
                 n_blocks: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Type[HybridDSCBasicBlockD] = HybridDSCBasicBlockD,
                 extend_scope: float = 1.0,
                 if_offset: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}

        self.conv_op = conv_op
        self.input_channels = input_channels
        self.output_channels = output_channels

        blocks = []
        for i in range(n_blocks):
            stride = initial_stride if i == 0 else 1
            in_planes = input_channels if i == 0 else output_channels
            
            blocks.append(block(
                in_planes, output_channels, conv_op, stride, conv_bias,
                norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, extend_scope, if_offset, device
            ))
            
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

    def compute_conv_feature_map_size(self, input_size):
        # Simplified computation
        output = 0
        for block in self.blocks:
            # Each block has 2 convs (or 1 conv + 1 DSC)
            output += np.prod([self.output_channels, *input_size], dtype=np.int64) * 2
        return output

# ========================================
# 3. DSC Residual Encoder
# ========================================

class DSCResidualEncoder(nn.Module):
    """
    DSC Enhanced Residual Encoder
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Type[HybridDSCBasicBlockD] = HybridDSCBasicBlockD,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 extend_scope: float = 1.0,
                 if_offset: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        # Parameter processing
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}

        # Build stem
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            # Use standard conv for stem (more stable)
            self.stem = nn.Sequential(
                conv_op(input_channels, stem_channels, kernel_sizes[0], 1, 
                       kernel_sizes[0]//2, bias=conv_bias),
                norm_op(stem_channels, **norm_op_kwargs) if norm_op else nn.Identity(),
                nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
            )
            input_channels = stem_channels
        else:
            self.stem = None

        # Build stages
        stages = []
        for s in range(n_stages):
            stage = DSCStackedResidualBlocks(
                n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s],
                kernel_sizes[s], strides[s], conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, block,
                extend_scope, if_offset, device
            )
            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips

        # Store parameters for decoder compatibility
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
            
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = np.prod([self.output_channels[0], *input_size], dtype=np.int64)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, 
                         [self.strides[s]] * len(input_size) if isinstance(self.strides[s], int) else self.strides[s])]

        return output

# ========================================
# 4. DSC Enhanced Decoder Components
# ========================================

class DSCTransposeConv(nn.Module):
    """
    DSC enhanced transpose convolution for upsampling
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 2,
                 padding: Union[int, Tuple[int, ...]] = 1,
                 output_padding: Union[int, Tuple[int, ...]] = 1,
                 conv_op: Type[_ConvNd] = nn.Conv2d,
                 bias: bool = True,
                 extend_scope: float = 1.0,
                 if_offset: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        self.is_3d = conv_op == nn.Conv3d
        
        if not self.is_3d:
            # 2D: Standard transpose conv + DSC enhancement
            transpconv_op = nn.ConvTranspose2d
            self.transpose_conv = transpconv_op(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
            
            # DSC enhancement for upsampled features
            self.dsc_enhance = DSCBlock(out_channels, out_channels, kernel_size=5, 
                                      extend_scope=extend_scope, if_offset=if_offset, device=device)
        else:
            # 3D: Standard transpose conv only
            transpconv_op = nn.ConvTranspose3d
            self.transpose_conv = transpconv_op(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
            self.dsc_enhance = None
    
    def forward(self, x):
        # Transpose convolution for upsampling
        x = self.transpose_conv(x)
        
        # DSC enhancement for 2D
        if self.dsc_enhance is not None:
            x_enhanced = self.dsc_enhance(x)
            # Residual connection
            x = x + x_enhanced
            
        return x

class DSCDecoderBlock(nn.Module):
    """
    DSC enhanced decoder block for feature fusion
    """
    def __init__(self,
                 n_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 extend_scope: float = 1.0,
                 if_offset: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}
            
        self.nonlin_first = nonlin_first
        self.is_3d = conv_op == nn.Conv3d
        
        # Build conv blocks
        blocks = []
        
        for i in range(n_convs):
            in_ch = input_channels if i == 0 else output_channels
            
            if not self.is_3d and i == (n_convs - 1):  # Last conv with DSC enhancement
                # DSC enhanced final conv
                blocks.append(DSCEnhancedConv(
                    in_ch, output_channels, kernel_size, conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs, nonlin_first, extend_scope, if_offset, device
                ))
            else:
                # Standard conv
                blocks.append(StandardConvBlock(
                    in_ch, output_channels, kernel_size, conv_op, conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                    nonlin, nonlin_kwargs, nonlin_first
                ))
        
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.blocks(x)

class DSCDeepSupervisionHead(nn.Module):
    """DSC enhanced segmentation head for deep supervision"""
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 extend_scope: float = 1.0,
                 if_offset: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        # DSC enhancement before segmentation
        self.dsc_enhance = DSCBlock(input_channels, input_channels, kernel_size=5,
                                  extend_scope=extend_scope, if_offset=if_offset, device=device)
        
        # Segmentation head
        self.seg_conv = nn.Conv2d(input_channels, num_classes, 1, 1, 0, bias=True)
        
        # Feature refinement
        self.refine_conv = nn.Conv2d(input_channels + input_channels, input_channels, 1, 1, 0)
        
    def forward(self, x):
        # DSC enhancement
        enhanced = self.dsc_enhance(x)
        
        # Feature refinement
        if enhanced.shape[2:] != x.shape[2:]:
            enhanced = F.adaptive_avg_pool2d(enhanced, x.shape[2:])
        
        refined = self.refine_conv(torch.cat([x, enhanced], dim=1))
        
        # Segmentation
        seg_out = self.seg_conv(refined)
        
        return seg_out

# ========================================
# 5. Complete DSC Enhanced Decoder
# ========================================

class DSCUNetDecoder(nn.Module):
    """
    Complete DSC enhanced UNet decoder
    """
    def __init__(self,
                 encoder,  # DSCResidualEncoder
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None,
                 extend_scope: float = 1.0,
                 if_offset: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.device = device
        
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        # Get parameters from encoder
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        # Build decoder stages
        stages = []
        transpconvs = []
        seg_layers = []
        
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            
            # DSC enhanced transpose convolution
            transpconvs.append(DSCTransposeConv(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                conv_op=encoder.conv_op, bias=conv_bias, extend_scope=extend_scope, 
                if_offset=if_offset, device=device
            ))
            
            # DSC enhanced decoder block
            stages.append(DSCDecoderBlock(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first,
                extend_scope, if_offset, device
            ))

            # DSC enhanced segmentation heads
            if encoder.conv_op == nn.Conv2d:  # Only for 2D
                seg_layers.append(DSCDeepSupervisionHead(
                    input_features_skip, num_classes, extend_scope, if_offset, device
                ))
            else:  # 3D fallback
                seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        
        for s in range(len(self.stages)):
            # DSC enhanced upsampling
            x = self.transpconvs[s](lres_input)
            
            # Skip connection concatenation
            skip = skips[-(s+2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((x, skip), 1)
            
            # DSC enhanced feature fusion
            x = self.stages[s](x)
            
            # DSC enhanced segmentation
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            
            lres_input = x

        # Invert seg outputs
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        # Similar to original but with DSC considerations
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, 
                             [self.encoder.strides[s]] * len(input_size) if isinstance(self.encoder.strides[s], int) else self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        output = np.int64(0)
        for s in range(len(self.stages)):
            # Decoder blocks (with DSC enhancement factor)
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64) * 3  # DSC enhancement factor
            # Transpose conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # Segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        
        return output

# ========================================
# 6. Complete DSC Residual UNet
# ========================================

class CompleteDSCResidualUNet(nn.Module):
    """
    Complete DSC Residual UNet with both encoder and decoder enhancement
    Main class for tubular structure segmentation
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None,
                 extend_scope: float = 1.0,
                 if_offset: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        # Parameter validation
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # DSC Enhanced Encoder
        self.encoder = DSCResidualEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, HybridDSCBasicBlockD, return_skips=True,
            disable_default_stem=False, stem_channels=stem_channels,
            extend_scope=extend_scope, if_offset=if_offset, device=device
        )

        # DSC Enhanced Decoder
        self.decoder = DSCUNetDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
            nonlin_first=False, extend_scope=extend_scope, if_offset=if_offset, device=device
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        return self.encoder.compute_conv_feature_map_size(input_size) + \
               self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        """
        Initialize the network weights
        """
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        module.apply(init_weights)

# ========================================
# 7. Factory Functions
# ========================================

def create_dsc_residual_unet_2d(
    input_channels: int = 1,
    num_classes: int = 2,
    n_stages: int = 6,
    features_per_stage: Tuple[int, ...] = (32, 64, 128, 256, 320, 320),
    kernel_sizes: Union[int, Tuple[int, ...]] = 3,
    strides: Tuple[int, ...] = (1, 2, 2, 2, 2, 2),
    n_blocks_per_stage: Union[int, Tuple[int, ...]] = 2,
    n_conv_per_stage_decoder: Union[int, Tuple[int, ...]] = (2, 2, 2, 2, 2),
    deep_supervision: bool = True,
    extend_scope: float = 1.0,
    if_offset: bool = True,
    device: str = 'cuda'
) -> CompleteDSCResidualUNet:
    """
    Create DSC Residual UNet for 2D images
    """
    model = CompleteDSCResidualUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv2d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=False,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={},
        dropout_op=None,
        dropout_op_kwargs={},
        nonlin=nn.ReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
        extend_scope=extend_scope,
        if_offset=if_offset,
        device=device
    )
    
    return model

def create_dsc_residual_unet_3d(
    input_channels: int = 1,
    num_classes: int = 2,
    n_stages: int = 6,
    features_per_stage: Tuple[int, ...] = (32, 64, 128, 256, 320, 320),
    kernel_sizes: Union[int, Tuple[int, ...]] = 3,
    strides: Tuple[int, ...] = (1, 2, 2, 2, 2, 2),
    n_blocks_per_stage: Union[int, Tuple[int, ...]] = 2,
    n_conv_per_stage_decoder: Union[int, Tuple[int, ...]] = (2, 2, 2, 2, 2),
    deep_supervision: bool = True,
    device: str = 'cuda'
) -> CompleteDSCResidualUNet:
    """
    Create DSC Residual UNet for 3D images (DSC disabled, uses standard conv)
    """
    model = CompleteDSCResidualUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=False,
        norm_op=nn.BatchNorm3d,
        norm_op_kwargs={},
        dropout_op=None,
        dropout_op_kwargs={},
        nonlin=nn.ReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
        extend_scope=1.0,  # Not used for 3D
        if_offset=True,    # Not used for 3D
        device=device
    )
    
    return model

# ========================================
# 8. Usage Examples and Testing
# ========================================

def test_dsc_models():
    """
    Test function for DSC models
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # Test 2D model
    print("\n=== Testing 2D DSC Residual UNet ===")
    model_2d = create_dsc_residual_unet_2d(
        input_channels=1,
        num_classes=3,  # NG tube segmentation: background, tube, tip
        deep_supervision=True,
        extend_scope=1.0,
        if_offset=True,
        device=device
    )
    
    # Initialize and move to device
    model_2d.initialize(model_2d)
    model_2d = model_2d.to(device)
    
    # Test forward pass
    x_2d = torch.randn(2, 1, 128, 128).to(device)
    
    try:
        with torch.no_grad():
            output_2d = model_2d(x_2d)
            print(f"2D Input shape: {x_2d.shape}")
            if isinstance(output_2d, list):
                print("2D Deep supervision outputs:")
                for i, out in enumerate(output_2d):
                    print(f"  Level {i}: {out.shape}")
            else:
                print(f"2D Output shape: {output_2d.shape}")
            print("2D Model test: SUCCESS!")
    except Exception as e:
        print(f"2D Model test: FAILED - {e}")
    
    # Test 3D model
    print("\n=== Testing 3D DSC Residual UNet ===")
    model_3d = create_dsc_residual_unet_3d(
        input_channels=1,
        num_classes=3,
        deep_supervision=True,
        device=device
    )
    
    # Initialize and move to device
    model_3d.initialize(model_3d)
    model_3d = model_3d.to(device)
    
    # Test forward pass
    x_3d = torch.randn(1, 1, 64, 64, 64).to(device)
    
    try:
        with torch.no_grad():
            output_3d = model_3d(x_3d)
            print(f"3D Input shape: {x_3d.shape}")
            if isinstance(output_3d, list):
                print("3D Deep supervision outputs:")
                for i, out in enumerate(output_3d):
                    print(f"  Level {i}: {out.shape}")
            else:
                print(f"3D Output shape: {output_3d.shape}")
            print("3D Model test: SUCCESS!")
    except Exception as e:
        print(f"3D Model test: FAILED - {e}")
    
    # Memory usage
    print(f"\n=== Model Information ===")
    total_params_2d = sum(p.numel() for p in model_2d.parameters())
    total_params_3d = sum(p.numel() for p in model_3d.parameters())
    print(f"2D Model total parameters: {total_params_2d:,}")
    print(f"3D Model total parameters: {total_params_3d:,}")
    
    return model_2d, model_3d

if __name__ == '__main__':
    print("=" * 60)
    print("Complete DSC Residual UNet Implementation")
    print("nnU-Net Compatible | Tubular Structure Optimized")
    print("=" * 60)
    
    # Run tests
    model_2d, model_3d = test_dsc_models()
    
    print("\n" + "=" * 60)
    print("DSC Residual UNet implementation completed!")
    print("Ready for nnU-Net training pipeline integration.")
    print("=" * 60)
    
    # Example usage for NG tube segmentation
    print("\n=== Example: NG Tube Segmentation Setup ===")
    ng_tube_model = create_dsc_residual_unet_2d(
        input_channels=1,           # Grayscale medical images
        num_classes=3,              # Background, NG tube body, NG tube tip
        n_stages=6,                 # Standard nnU-Net stages
        deep_supervision=True,      # Better gradient flow
        extend_scope=1.0,           # DSC deformation range
        if_offset=True,             # Enable adaptive deformation
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"NG Tube Model created with {sum(p.numel() for p in ng_tube_model.parameters()):,} parameters")
    print("Model ready for training with nnU-Net pipeline!")
    print(ng_tube_model)
