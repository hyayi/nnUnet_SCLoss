import torch
import torch.nn as nn
import torch.nn.functional as F  # F (functional) 임포트
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
# 1. Dynamic Snake Convolution (수정된 버전)
# - DSC 클래스 제거, DSConv 모듈로 통합
# - for 루프 -> torch.cumsum
# - 수동 interpolate -> F.grid_sample
# =====================================================================================

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
        
        # DSC 헬퍼 클래스의 속성들을 DSConv가 직접 저장
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = device

    def forward(self, f):
        B, C, W, H = f.shape  # ✅ 입력 텐서에서 동적으로 Shape 가져오기
        offset = self.offset_conv(f)
        offset = torch.tanh(self.bn(offset))
        
        # 🔥 DSC 객체 생성 없이 헬퍼 메서드 직접 호출
        deformed_feature = self._deform_conv(f, offset, B, C, W, H)
        
        x = self.dsc_conv_x(deformed_feature) if self.morph == 0 else self.dsc_conv_y(deformed_feature)
        return self.relu(self.gn(x))

    def _deform_conv(self, input_tensor, offset, B, C, W, H):
        y, x = self._coordinate_map_3D(offset, B, W, H)
        # 🔥 최적화된 grid_sample 함수 호출
        return self._bilinear_interpolate_with_grid_sample(input_tensor, y, x, B, C, W, H)

    def _coordinate_map_3D(self, offset, B, W, H):
        y_offset, x_offset = torch.split(offset, self.kernel_size, dim=1)

        y_center = torch.arange(0, W, device=self.device).repeat(H).reshape(H, W).permute(1, 0).reshape(-1, W, H).repeat(self.kernel_size, 1, 1).float().unsqueeze(0)
        x_center = torch.arange(0, H, device=self.device).repeat(W).reshape(W, H).reshape(-1, W, H).repeat(self.kernel_size, 1, 1).float().unsqueeze(0)

        if self.morph == 0:
            y = torch.linspace(0, 0, 1, device=self.device)
            x = torch.linspace(-int(self.kernel_size // 2), int(self.kernel_size // 2), self.kernel_size, device=self.device)
            y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
            y_grid = y_grid.reshape(-1, 1).repeat(1, W * H).reshape(self.kernel_size, W, H).unsqueeze(0)
            x_grid = x_grid.reshape(-1, 1).repeat(1, W * H).reshape(self.kernel_size, W, H).unsqueeze(0)

            y_new = (y_center + y_grid).repeat(B, 1, 1, 1)
            x_new = (x_center + x_grid).repeat(B, 1, 1, 1)

            if self.if_offset:
                # 🔥 for 루프를 torch.cumsum으로 대체 (속도 향상)
                y_offset_permuted = y_offset.permute(1, 0, 2, 3) # [K, B, W, H]
                center = self.kernel_size // 2

                y_offset_fwd = torch.cumsum(y_offset_permuted[center:], dim=0)
                y_offset_rev_flipped = torch.cumsum(torch.flip(y_offset_permuted[:center], dims=[0]), dim=0)
                y_offset_rev = torch.flip(y_offset_rev_flipped, dims=[0])
                
                y_offset_fwd[0] = 0 # Center offset은 0
                y_offset_new = torch.cat((y_offset_rev, y_offset_fwd), dim=0) # [K, B, W, H]
                
                y_new = y_new.add(y_offset_new.permute(1, 0, 2, 3).mul(self.extend_scope))

            y_new = y_new.reshape(B, self.kernel_size, 1, W, H).permute(0, 3, 1, 4, 2).reshape(B, self.kernel_size * W, H)
            x_new = x_new.reshape(B, self.kernel_size, 1, W, H).permute(0, 3, 1, 4, 2).reshape(B, self.kernel_size * W, H)
            return y_new, x_new
        
        else: # morph == 1
            y = torch.linspace(-int(self.kernel_size // 2), int(self.kernel_size // 2), self.kernel_size, device=self.device)
            x = torch.linspace(0, 0, 1, device=self.device)
            y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
            y_grid = y_grid.reshape(-1, 1).repeat(1, W * H).reshape(self.kernel_size, W, H).unsqueeze(0)
            x_grid = x_grid.reshape(-1, 1).repeat(1, W * H).reshape(self.kernel_size, W, H).unsqueeze(0)

            y_new = (y_center + y_grid).repeat(B, 1, 1, 1)
            x_new = (x_center + x_grid).repeat(B, 1, 1, 1)
            
            if self.if_offset:
                # 🔥 for 루프를 torch.cumsum으로 대체 (속도 향상)
                x_offset_permuted = x_offset.permute(1, 0, 2, 3) # [K, B, W, H]
                center = self.kernel_size // 2
                
                x_offset_fwd = torch.cumsum(x_offset_permuted[center:], dim=0)
                x_offset_rev_flipped = torch.cumsum(torch.flip(x_offset_permuted[:center], dims=[0]), dim=0)
                x_offset_rev = torch.flip(x_offset_rev_flipped, dims=[0])

                x_offset_fwd[0] = 0 # Center offset은 0
                x_offset_new = torch.cat((x_offset_rev, x_offset_fwd), dim=0) # [K, B, W, H]

                x_new = x_new.add(x_offset_new.permute(1, 0, 2, 3).mul(self.extend_scope))
            
            y_new = y_new.reshape(B, 1, self.kernel_size, W, H).permute(0, 3, 1, 4, 2).reshape(B, W, self.kernel_size * H)
            x_new = x_new.reshape(B, 1, self.kernel_size, W, H).permute(0, 3, 1, 4, 2).reshape(B, W, self.kernel_size * H)
            return y_new, x_new

    def _bilinear_interpolate_with_grid_sample(self, input_feature, y, x, B, C, W, H):
        # input_feature: [B, C, W, H]
        # y: [B, K*W, H] (morph=0) or [B, W, K*H] (morph=1)
        # x: [B, K*W, H] (morph=0) or [B, W, K*H] (morph=1)
        
        # ⚠️ nan_to_num 제거! NaN 발생 시 학습이 중단되어야 원인을 찾을 수 있음.

        # 🔥 F.grid_sample을 위한 좌표 정규화 ([-1, 1] 범위로)
        # grid_sample의 'x' 좌표 (W_out 차원)는 H (height)를 기준으로,
        # 'y' 좌표 (H_out 차원)는 W (width)를 기준으로 정규화합니다.
        y_norm = (y / (W - 1)) * 2 - 1
        x_norm = (x / (H - 1)) * 2 - 1

        # F.grid_sample은 [B, H_out, W_out, 2] 형태의 grid를 기대합니다.
        if self.morph == 0:
            # y, x shape: [B, K*W, H]
            # H_out = K*W, W_out = H
            grid = torch.stack((x_norm, y_norm), dim=-1) # Shape: [B, K*W, H, 2]
        else: # morph == 1
            # y, x shape: [B, W, K*H]
            # H_out = W, W_out = K*H
            grid = torch.stack((x_norm, y_norm), dim=-1) # Shape: [B, W, K*H, 2]

        # 🔥 F.grid_sample 실행
        outputs = F.grid_sample(
            input_feature, 
            grid, 
            mode='bilinear', 
            padding_mode='border', # 'border'는 clamp와 유사하게 동작
            align_corners=True     # 수동 구현이 (0,0) (W-1, H-1)을 기준으로 했으므로 True
        )
        
        # outputs shape은 [B, C, H_out, W_out]이므로,
        # morph=0 -> [B, C, K*W, H]
        # morph=1 -> [B, C, W, K*H]
        # 이는 원본 코드의 최종 출력 형태와 일치하므로 추가 permute/reshape 불필요.
        return outputs

# =====================================================================================
# 2. DSCUNet 빌딩 블록 (변경 사항 없음)
# - DSConv의 __init__ 시그니처가 동일하므로 이 섹션은 수정할 필요가 없습니다.
# =====================================================================================

class DSCBlock(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, kernel_size, initial_stride, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, device):
        super().__init__()
        self.conv_standard = StackedConvBlocks(1, conv_op, input_channels, output_channels, kernel_size, initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first)
        # ✅ 수정된 DSConv 호출 (파라미터는 동일)
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
# 3. nnU-Net Encoder와 Decoder (변경 사항 없음)
# =====================================================================================

class DSCEncoder(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips, nonlin_first, pool, device):
        super().__init__()
        self.device = device # DSCDecoder가 접근할 수 있도록 self.device에 저장
        
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
            
            # ✅ 수정된 StackedDSCBlocks 호출 (파라미터는 동일)
            stage_modules.append(StackedDSCBlocks(n_conv_per_stage[s], conv_op, current_in_channels, features_per_stage[s], kernel_sizes[s], conv_stride, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, device))
            stages.append(nn.Sequential(*stage_modules))
            current_in_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
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
    def __init__(self,
                 encoder: DSCEncoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 conv_bias: bool = None,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op, norm_op_kwargs = (encoder.norm_op, encoder.norm_op_kwargs) if norm_op is None else (norm_op, norm_op_kwargs)
        dropout_op, dropout_op_kwargs = (encoder.dropout_op, encoder.dropout_op_kwargs) if dropout_op is None else (dropout_op, dropout_op_kwargs)
        nonlin, nonlin_kwargs = (encoder.nonlin, encoder.nonlin_kwargs) if nonlin is None else (nonlin, nonlin_kwargs)
        device = encoder.device # ✅ Encoder로부터 device 정보 가져오기

        self.stages = nn.ModuleList()
        self.transpconvs = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        for s in range(1, n_stages_encoder):
            input_features_below, input_features_skip = encoder.output_channels[-s], encoder.output_channels[-(s + 1)]
            self.transpconvs.append(transpconv_op(input_features_below, input_features_skip, encoder.strides[-s], encoder.strides[-s], bias=conv_bias))
            
            # ✅ 수정된 StackedDSCBlocks 호출 (파라미터는 동일)
            self.stages.append(StackedDSCBlocks(n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip, encoder.kernel_sizes[-(s+1)], 1, encoder.dsc_kernel_size, encoder.dsc_extend_scope, encoder.dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first, device))
            
            self.seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
                
            lres_input = x
        
        seg_outputs = seg_outputs[::-1]
        
        return seg_outputs if self.deep_supervision else seg_outputs[0]

# =====================================================================================
# 4. 최종 DSCUNet 모델 (변경 사항 없음)
# =====================================================================================

class DSCUNet(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, num_classes, n_conv_per_stage_decoder, n_conv_per_stage=None, n_blocks_per_stage=None, dsc_kernel_size=9, dsc_extend_scope=1, dsc_if_offset=True, conv_bias=False, norm_op=None, norm_op_kwargs=None, dropout_op=None, dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None, deep_supervision=False, nonlin_first=False, device='cuda'):
        super().__init__()
        
        if n_conv_per_stage is None and n_blocks_per_stage is None:
            raise ValueError("Must provide either 'n_conv_per_stage' or 'n_blocks_per_stage'.")
        _n_conv_per_stage_encoder = n_conv_per_stage if n_conv_per_stage is not None else n_blocks_per_stage
        
        if conv_op is not nn.Conv2d:
            warnings.warn("DSCUNet is designed for 2D (nn.Conv2d) only.")
        
        # ✅ 수정된 DSCEncoder 호출 (파라미터는 동일)
        self.encoder = DSCEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, _n_conv_per_stage_encoder, dsc_kernel_size, dsc_extend_scope, dsc_if_offset, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, True, nonlin_first, 'conv', device)
        self.decoder = DSCDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

# =====================================================================================
# 5. 테스트 코드 (⚠️ 디버깅 코드 추가)
# =====================================================================================

if __name__ == '__main__':
    # ⚠️ NaN/Inf 발생 시 즉시 오류를 발생시켜 원인을 추적합니다.
    # ⚠️ 성능이 오르지 않는 문제를 디버깅하기 위해 꼭 필요합니다.
    torch.autograd.set_detect_anomaly(True)

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

    print("\nForward pass 실행 중 (anomaly detection 활성화됨)...")
    # torch.no_grad()를 사용하면 역전파를 안하므로 anomaly detection이
    # 큰 의미가 없을 수 있습니다. 실제 학습 시 (model.train() 및 loss.backward())에
    # anomaly detection이 진가를 발휘합니다.
    # 여기서는 순전파 자체의 오류를 잡기 위해 실행합니다.
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
    
    print("\n---")
    print("💡 참고: `torch.autograd.set_detect_anomaly(True)`가 활성화되었습니다.")
    print("   실제 학습(loss.backward()) 중에 NaN이 발생하면 프로그램이 중단되고")
    print("   오류 추적 정보(traceback)가 출력될 것입니다.")
    print("\n   만약 `NaN`으로 인한 오류가 계속 발생한다면,")
    print("   1. Gradient Clipping을 적용해 보세요 (예: `torch.nn.utils.clip_grad_norm_`)")
    print("   2. `DSConv._coordinate_map_3D`에서 `cumsum` 이후 `tanh`를 추가하여 offset 범위를 제한해 보세요.")
    print("---")
