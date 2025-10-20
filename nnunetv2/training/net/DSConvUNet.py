import torch
import torch.nn as nn
import numpy as np
from typing import Union, Type, List, Tuple
import warnings

# PyTorch 및 dynamic_network_architectures 라이브러리에서 필요한 모듈 임포트
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp, maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

warnings.filterwarnings("ignore")

# =====================================================================================
# 1. Dynamic Snake Convolution 원본 코드 (인덱싱 버그가 수정된 최종 버전)
# =====================================================================================

class DSC(object):
    """
    DSConv의 핵심 변형 연산을 수행하는 헬퍼 클래스.
    - NaN/inf 값으로 인한 인덱싱 오류를 방지하는 안전장치 포함.
    """
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    def _coordinate_map_3D(self, offset, if_offset):
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)
        
        y_center = torch.arange(0, self.width, device=self.device).repeat(self.height).reshape(self.height, self.width).permute(1, 0).reshape(-1, self.width, self.height).repeat(self.num_points, 1, 1).float().unsqueeze(0)
        x_center = torch.arange(0, self.height, device=self.device).repeat(self.width).reshape(self.width, self.height).reshape(-1, self.width, self.height).repeat(self.num_points, 1, 1).float().unsqueeze(0)

        if self.morph == 0:
            y = torch.linspace(0, 0, 1, device=self.device)
            x = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), self.num_points, device=self.device)
            y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
            y_grid = y_grid.reshape(-1, 1).repeat(1, self.width * self.height).reshape(self.num_points, self.width, self.height).unsqueeze(0)
            x_grid = x_grid.reshape(-1, 1).repeat(1, self.width * self.height).reshape(self.num_points, self.width, self.height).unsqueeze(0)
            
            y_new = (y_center + y_grid).repeat(self.num_batch, 1, 1, 1)
            x_new = (x_center + x_grid).repeat(self.num_batch, 1, 1, 1)

            if if_offset:
                y_offset_new = y_offset.detach().clone().permute(1, 0, 2, 3)
                y_offset_permuted = y_offset.permute(1, 0, 2, 3)
                center = self.num_points // 2
                y_offset_new[center] = 0
                for index in range(1, center + 1):
                    if center + index < y_offset_new.shape[0]:
                        y_offset_new[center + index] = y_offset_new[center + index - 1] + y_offset_permuted[center + index]
                    if center - index >= 0:
                        y_offset_new[center - index] = y_offset_new[center - index + 1] + y_offset_permuted[center - index]
                y_new = y_new.add(y_offset_new.permute(1, 0, 2, 3).mul(self.extend_scope))

            y_new = y_new.reshape(self.num_batch, self.num_points, 1, self.width, self.height).permute(0, 3, 1, 4, 2).reshape(self.num_batch, self.num_points * self.width, self.height)
            x_new = x_new.reshape(self.num_batch, self.num_points, 1, self.width, self.height).permute(0, 3, 1, 4, 2).reshape(self.num_batch, self.num_points * self.width, self.height)
            return y_new, x_new
        else: # morph == 1
            y = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), self.num_points, device=self.device)
            x = torch.linspace(0, 0, 1, device=self.device)
            y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
            y_grid = y_grid.reshape(-1, 1).repeat(1, self.width * self.height).reshape(self.num_points, self.width, self.height).unsqueeze(0)
            x_grid = x_grid.reshape(-1, 1).repeat(1, self.width * self.height).reshape(self.num_points, self.width, self.height).unsqueeze(0)

            y_new = (y_center + y_grid).repeat(self.num_batch, 1, 1, 1)
            x_new = (x_center + x_grid).repeat(self.num_batch, 1, 1, 1)
            
            if if_offset:
                x_offset_new = x_offset.detach().clone().permute(1, 0, 2, 3)
                x_offset_permuted = x_offset.permute(1, 0, 2, 3)
                center = self.num_points // 2
                x_offset_new[center] = 0
                for index in range(1, center + 1):
                    if center + index < x_offset_new.shape[0]:
                        x_offset_new[center + index] = x_offset_new[center + index - 1] + x_offset_permuted[center + index]
                    if center - index >= 0:
                        x_offset_new[center - index] = x_offset_new[center - index + 1] + x_offset_permuted[center - index]
                x_new = x_new.add(x_offset_new.permute(1, 0, 2, 3).mul(self.extend_scope))
            
            y_new = y_new.reshape(self.num_batch, 1, self.num_points, self.width, self.height).permute(0, 3, 1, 4, 2).reshape(self.num_batch, self.width, self.num_points * self.height)
            x_new = x_new.reshape(self.num_batch, 1, self.num_points, self.width, self.height).permute(0, 3, 1, 4, 2).reshape(self.num_batch, self.width, self.num_points * self.height)
            return y_new, x_new

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        # 1. NaN/inf를 안전한 값 0으로 변환
        y = torch.nan_to_num(y.reshape([-1]).float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.nan_to_num(x.reshape([-1]).float(), nan=0.0, posinf=0.0, neginf=0.0)

        zero = torch.tensor(0, device=self.device, dtype=torch.int32)
        max_y, max_x = self.width - 1, self.height - 1

        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # 2. 인덱싱에 사용할 최종 좌표를 유효한 범위로 강제 제한 (clamping)
        y0, y1 = torch.clamp(y0, zero, max_y), torch.clamp(y1, zero, max_y)
        x0, x1 = torch.clamp(x0, zero, max_x), torch.clamp(x1, zero, max_x)
        
        input_feature_flat = input_feature.permute(0, 2, 3, 1).reshape(-1, self.num_channels)
        dimension = self.height * self.width
        base = (torch.arange(self.num_batch, device=self.device) * dimension).reshape(-1, 1).float()
        repeat = torch.ones([y.shape[0] // self.num_batch], device=self.device).unsqueeze(0).float()
        base = torch.matmul(base, repeat).reshape(-1)

        base_y0, base_y1 = (base + (y0 * self.height).long()), (base + (y1 * self.height).long())
        idx_a, idx_b = (base_y0 + x0).long(), (base_y1 + x0).long()
        idx_c, idx_d = (base_y0 + x1).long(), (base_y1 + x1).long()
        
        max_idx = input_feature_flat.shape[0] - 1
        idx_a, idx_b = torch.clamp(idx_a, 0, max_idx), torch.clamp(idx_b, 0, max_idx)
        idx_c, idx_d = torch.clamp(idx_c, 0, max_idx), torch.clamp(idx_d, 0, max_idx)

        value_a, value_b = input_feature_flat[idx_a], input_feature_flat[idx_b]
        value_c, value_d = input_feature_flat[idx_c], input_feature_flat[idx_d]

        vol_a = ((y1.float() - y) * (x1.float() - x)).unsqueeze(-1)
        vol_b = ((y - y0.float()) * (x1.float() - x)).unsqueeze(-1)
        vol_c = ((y1.float() - y) * (x - x0.float())).unsqueeze(-1)
        vol_d = ((y - y0.float()) * (x - x0.float())).unsqueeze(-1)

        outputs = value_a * vol_a + value_b * vol_b + value_c * vol_c + value_d * vol_d

        if self.morph == 0:
            outputs = outputs.reshape(self.num_batch, self.num_points * self.width, self.height, self.num_channels).permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape(self.num_batch, self.width, self.num_points * self.height, self.num_channels).permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input_tensor, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        return self._bilinear_interpolate_3D(input_tensor, y, x)

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset, device):
        super(DSConv, self).__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size
        self.dsc_conv_x = nn.Conv2d(in_ch, out_ch, (kernel_size, 1), stride=(kernel_size, 1), padding=0)
        self.dsc_conv_y = nn.Conv2d(in_ch, out_ch, (1, kernel_size), stride=(1, kernel_size), padding=0)
        self.gn = nn.GroupNorm(out_ch // 4 if out_ch > 1 and out_ch % 4 == 0 else 1, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.extend_scope, self.morph, self.if_offset, self.device = extend_scope, morph, if_offset, device

    def forward(self, f):
        offset = self.offset_conv(f)
        offset = torch.tanh(self.bn(offset))
        dsc = DSC(f.shape, self.kernel_size, self.extend_scope, self.morph, self.device)
        deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
        x = self.dsc_conv_x(deformed_feature) if self.morph == 0 else self.dsc_conv_y(deformed_feature)
        return self.relu(self.gn(x))

# =====================================================================================
# 2. DSCUNet 빌딩 블록 (모든 파라미터 이름 및 전달 오류 수정됨)
# =====================================================================================

class DSCBlock(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, kernel_size, initial_stride, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, device):
        super().__init__()
        self.conv_standard = StackedConvBlocks(1, conv_op, input_channels, output_channels, kernel_size, initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first)
        self.dsc_x = DSConv(input_channels, output_channels, dsc_kernel_size, dsc_extend_scope, 0, dsc_if_offset, device)
        self.dsc_y = DSConv(input_channels, output_channels, dsc_kernel_size, dsc_extend_scope, 1, dsc_if_offset, device)
        self.conv_merge = StackedConvBlocks(1, conv_op, output_channels * 3, output_channels, 1, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first)

    def forward(self, x):
        out_std = self.conv_standard(x)
        out_dsc_x = self.dsc_x(x)
        out_dsc_y = self.dsc_y(x)
        target_size = out_std.shape[2:]
        out_dsc_x = nn.functional.interpolate(out_dsc_x, size=target_size, mode='bilinear', align_corners=False)
        out_dsc_y = nn.functional.interpolate(out_dsc_y, size=target_size, mode='bilinear', align_corners=False)
        combined = torch.cat((out_std, out_dsc_x, out_dsc_y), dim=1)
        return self.conv_merge(combined)

class StackedDSCBlocks(nn.Module):
    def __init__(self, num_convs, conv_op, input_channels, output_channels, kernel_size, initial_stride, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, device):
        super().__init__()
        self.convs = nn.Sequential(
            DSCBlock(conv_op, input_channels, output_channels, kernel_size, initial_stride, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, device),
            *[DSCBlock(conv_op, output_channels, output_channels, kernel_size, 1, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, device) for _ in range(num_convs - 1)]
        )
    def forward(self, x):
        return self.convs(x)

# =====================================================================================
# 3. nnU-Net Encoder와 Decoder (모든 오류 수정됨)
# =====================================================================================

class DSCEncoder(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips, nonlin_first, pool, device):
        super().__init__()
        self.device = device # DSCDecoder가 접근할 수 있도록 self.device에 저장
        
        # ... (나머지 __init__ 코드는 이전과 동일)
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int): features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int): n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int): strides = [strides] * n_stages
        
        stages = []
        current_in_channels = input_channels
        for s in range(n_stages):
            conv_stride = strides[s] if pool == 'conv' else 1
            stage_modules = []
            if pool in ['max', 'avg'] and any(i != 1 for i in (strides[s] if isinstance(strides[s], (tuple, list)) else [strides[s]])):
                stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
            
            stage_modules.append(StackedDSCBlocks(n_conv_per_stage[s], conv_op, current_in_channels, features_per_stage[s], kernel_sizes[s], conv_stride, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, device))
            stages.append(nn.Sequential(*stage_modules))
            current_in_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages) # Sequential 대신 ModuleList로 변경하여 forward 수정
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.conv_op, self.norm_op, self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs, self.dropout_op, self.dropout_op_kwargs, self.conv_bias, self.kernel_sizes, self.dsc_kernel_size, self.dsc_extend_scope, self.dsc_if_offset = conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs, conv_bias, kernel_sizes, dsc_kernel_size, dsc_extend_scope, dsc_if_offset

    def forward(self, x):
        skips = []
        for s in self.stages:
            x = s(x)
            skips.append(x)
        return skips if self.return_skips else skips[-1]

class DSCDecoder(nn.Module):
    def __init__(self, encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first=False, conv_bias=None, norm_op=None, norm_op_kwargs=None, dropout_op=None, dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int): n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op, norm_op_kwargs = (encoder.norm_op, encoder.norm_op_kwargs) if norm_op is None else (norm_op, norm_op_kwargs)
        dropout_op, dropout_op_kwargs = (encoder.dropout_op, encoder.dropout_op_kwargs) if dropout_op is None else (dropout_op, dropout_op_kwargs)
        nonlin, nonlin_kwargs = (encoder.nonlin, encoder.nonlin_kwargs) if nonlin is None else (nonlin, nonlin_kwargs)
        device = encoder.device

        self.stages, self.transpconvs, self.seg_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for s in range(1, n_stages_encoder):
            input_features_below, input_features_skip = encoder.output_channels[-s], encoder.output_channels[-(s + 1)]
            self.transpconvs.append(transpconv_op(input_features_below, input_features_skip, encoder.strides[-s], encoder.strides[-s], bias=conv_bias))
            self.stages.append(StackedDSCBlocks(n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip, encoder.kernel_sizes[-(s+1)], 1, encoder.dsc_kernel_size, encoder.dsc_extend_scope, encoder.dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, device))
            if self.deep_supervision and s < (n_stages_encoder - 1):
                self.seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
        
        self.final_seg_layer = encoder.conv_op(encoder.output_channels[0], num_classes, 1, 1, 0, bias=True)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision and s < len(self.seg_layers):
                seg_outputs.append(self.seg_layers[s](x))
            lres_input = x
        
        seg_outputs.reverse()
        seg_outputs.insert(0, self.final_seg_layer(lres_input))
        return seg_outputs if self.deep_supervision else seg_outputs[0]

# =====================================================================================
# 4. 최종 DSCUNet 모델 (유연한 파라미터 처리 및 device 전달 기능 포함)
# =====================================================================================

class DSCUNet(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, num_classes, n_conv_per_stage_decoder, n_conv_per_stage=None, n_blocks_per_stage=None, dsc_kernel_size=9, dsc_extend_scope=1, dsc_if_offset=True, conv_bias=False, norm_op=None, norm_op_kwargs=None, dropout_op=None, dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None, deep_supervision=False, nonlin_first=False, device='cuda'):
        super().__init__()
        
        if n_conv_per_stage is None and n_blocks_per_stage is None:
            raise ValueError("Must provide either 'n_conv_per_stage' or 'n_blocks_per_stage'.")
        _n_conv_per_stage_encoder = n_conv_per_stage if n_conv_per_stage is not None else n_blocks_per_stage
        
        if conv_op is not nn.Conv2d:
            warnings.warn("DSCUNet is designed for 2D (nn.Conv2d) only.")
        
        self.encoder = DSCEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, _n_conv_per_stage_encoder, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, True, nonlin_first, 'conv', device)
        self.decoder = DSCDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

# =====================================================================================
# 5. 테스트 코드
# =====================================================================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 DSCUNet 모델 테스트를 시작합니다. 사용 장치: {device}")

    data = torch.rand((1, 1, 512, 512)).to(device)

    model = DSCUNet(
        input_channels=1,
        n_stages=8,
        features_per_stage=(32, 64, 128, 256, 512, 512, 512, 512),
        conv_op=nn.Conv2d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2, 2, 2, 2),
        n_conv_per_stage=2, # 또는 n_blocks_per_stage=2
        num_classes=4,
        n_conv_per_stage_decoder=2,
        dsc_kernel_size=9,
        dsc_extend_scope=1,
        dsc_if_offset=True,
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'negative_slope': 1e-2, 'inplace': True},
        deep_supervision=True,
        device=device
    ).to(device)

    DSCUNet.initialize(model)

    print("\nForward pass 실행 중...")
    with torch.no_grad():
        outputs = model(data)
    print("Forward pass 완료!")

    if isinstance(outputs, list):
        print(f"\nDeep Supervision 출력 ({len(outputs)}개):")
        for i, out in enumerate(outputs):
            print(f"  - 출력 {i} shape: {out.shape}")
    else:
        print(f"\n단일 출력 shape: {outputs.shape}")

    final_output_shape = outputs[0].shape
    expected_shape = (data.shape[0], 4, data.shape[2], data.shape[3])
    
    print(f"\n최종 출력 Shape: {final_output_shape}")
    print(f"예상 출력 Shape: {expected_shape}")
    
    assert final_output_shape == expected_shape, "최종 출력 shape가 예상과 다릅니다!"
    print("\n✅ DSCUNet 모델 테스트 성공!")
