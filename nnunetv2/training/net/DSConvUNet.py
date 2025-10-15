# nnunetv2/nets/DSConvUNet.py
from typing import Union, Type, List, Tuple
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from S3_DSConv_pro import DSConv_pro


class DSConvBlock(nn.Module):
    """
    DSConv Multi-View Block (원본 DSCNet 스타일)
    일반 conv + x축 DSConv + y축 DSConv → fusion
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, 
                 extend_scope=1.0, if_offset=True, device='cuda'):
        super().__init__()
        
        # 1) 일반 convolution branch
        self.conv_standard = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn_standard = nn.GroupNorm(out_channels // 4, out_channels)
        
        # 2) x축 방향 DSConv branch
        self.conv_x = DSConv_pro(
            in_channels, out_channels, kernel_size, extend_scope, 
            morph=0, if_offset=if_offset, device=device
        )
        
        # 3) y축 방향 DSConv branch  
        self.conv_y = DSConv_pro(
            in_channels, out_channels, kernel_size, extend_scope,
            morph=1, if_offset=if_offset, device=device
        )
        
        # 4) Fusion layer (3 branches → final output)
        self.fusion_conv = nn.Conv2d(3 * out_channels, out_channels, 3, padding=1)
        self.fusion_gn = nn.GroupNorm(out_channels // 4, out_channels)
        
        # 5) Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 3개 branch 병렬 실행
        out_standard = self.relu(self.gn_standard(self.conv_standard(x)))
        out_x = self.conv_x(x)  # DSConv는 내부에서 처리
        out_y = self.conv_y(x)  # DSConv는 내부에서 처리
        
        # Concatenate and fusion
        concatenated = torch.cat([out_standard, out_x, out_y], dim=1)
        fused = self.relu(self.fusion_gn(self.fusion_conv(concatenated)))
        
        return fused


class StandardConvBlock(nn.Module):
    """
    일반 convolution block (DSConv 없는 구간용)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class EncoderStage(nn.Module):
    """
    Encoder Stage = Conv blocks + Pooling (마지막 stage 제외)
    """
    def __init__(self, in_channels, out_channels, n_conv_blocks, 
                 use_dsconv=False, dsconv_kernel_size=9, extend_scope=1.0, 
                 if_offset=True, has_pooling=True):
        super().__init__()
        
        # Conv blocks
        self.conv_blocks = nn.ModuleList()
        
        for block_idx in range(n_conv_blocks):
            block_in_ch = in_channels if block_idx == 0 else out_channels
            
            if use_dsconv and block_idx == 0:
                # 첫 번째 블록만 DSConv 사용
                block = DSConvBlock(
                    block_in_ch, out_channels, 
                    kernel_size=dsconv_kernel_size,
                    extend_scope=extend_scope,
                    if_offset=if_offset,
                    device='cuda'
                )
            else:
                # 나머지는 일반 conv
                block = StandardConvBlock(block_in_ch, out_channels)
            
            self.conv_blocks.append(block)
        
        # Pooling (마지막 stage가 아닌 경우만)
        self.pool = nn.MaxPool2d(2) if has_pooling else None
    
    def forward(self, x):
        # Conv blocks 실행
        for block in self.conv_blocks:
            x = block(x)
        
        # Skip connection을 위해 pooling 전 feature 저장
        skip_feature = x
        
        # Pooling (있는 경우만)
        if self.pool is not None:
            x = self.pool(x)
        
        return x, skip_feature


class DecoderStage(nn.Module):
    """
    Decoder Stage = Upsample + Skip Connection + Conv blocks
    """
    def __init__(self, in_channels, skip_channels, out_channels, n_conv_blocks,
                 use_dsconv=True, dsconv_kernel_size=9, extend_scope=1.0, if_offset=True):
        super().__init__()
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Conv blocks (skip connection 후)
        self.conv_blocks = nn.ModuleList()
        concat_channels = in_channels + skip_channels  # upsampled + skip
        
        for block_idx in range(n_conv_blocks):
            block_in_ch = concat_channels if block_idx == 0 else out_channels
            
            if use_dsconv and block_idx == 0:
                # 첫 번째 블록은 DSConv
                block = DSConvBlock(
                    block_in_ch, out_channels,
                    kernel_size=dsconv_kernel_size,
                    extend_scope=extend_scope,
                    if_offset=if_offset,
                    device='cuda'
                )
            else:
                # 나머지는 일반 conv
                block = StandardConvBlock(block_in_ch, out_channels)
            
            self.conv_blocks.append(block)
    
    def forward(self, x, skip_feature):
        # Upsampling
        x = self.upsample(x)
        
        # Skip connection
        x = torch.cat([x, skip_feature], dim=1)
        
        # Conv blocks 실행
        for block in self.conv_blocks:
            x = block(x)
        
        return x


class DSConvUNet(nn.Module):
    """
    DSConvUNet - 각 stage에 pooling 포함된 깔끔한 버전
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
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
                 nonlin_first: bool = False,
                 # DSConv parameters
                 dsconv_kernel_size: int = 9,
                 extend_scope: float = 1.0,
                 if_offset: bool = True
                 ):
        super().__init__()
        
        # 파라미터 정규화
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * (2**i) for i in range(n_stages)]
        
        self.input_channels = input_channels
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        # DSConv 설정
        self.dsconv_kernel_size = dsconv_kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        
        # Encoder stages 구성 (pooling 포함)
        self.encoder_stages = nn.ModuleList()
        
        for stage_idx in range(n_stages):
            in_ch = input_channels if stage_idx == 0 else features_per_stage[stage_idx-1]
            out_ch = features_per_stage[stage_idx]
            
            # DSConv 사용 여부 (middle stages)
            use_dsconv = (1 <= stage_idx <= min(n_stages-2, 3))
            
            # 마지막 stage는 pooling 없음
            has_pooling = (stage_idx < n_stages - 1)
            
            encoder_stage = EncoderStage(
                in_channels=in_ch,
                out_channels=out_ch,
                n_conv_blocks=n_conv_per_stage[stage_idx],
                use_dsconv=use_dsconv,
                dsconv_kernel_size=dsconv_kernel_size,
                extend_scope=extend_scope,
                if_offset=if_offset,
                has_pooling=has_pooling
            )
            
            self.encoder_stages.append(encoder_stage)
        
        # Decoder stages 구성
        self.decoder_stages = nn.ModuleList()
        
        for stage_idx in range(n_stages - 1):
            # 채널 계산
            if stage_idx == 0:
                # 첫 번째 decoder: bottleneck → 이전 level
                in_ch = features_per_stage[-1]      # bottleneck (320)
                skip_ch = features_per_stage[-2]    # skip connection (256)
                out_ch = features_per_stage[-2]     # output (256)
            else:
                # 나머지 decoder stages
                in_ch = features_per_stage[-(stage_idx+1)]   # 이전 decoder 출력
                skip_ch = features_per_stage[-(stage_idx+2)] # skip connection
                out_ch = features_per_stage[-(stage_idx+2)]  # output
            
            decoder_stage = DecoderStage(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                n_conv_blocks=n_conv_per_stage_decoder[stage_idx],
                use_dsconv=True,  # decoder에서는 무조건 DSConv
                dsconv_kernel_size=dsconv_kernel_size,
                extend_scope=extend_scope,
                if_offset=if_offset
            )
            
            self.decoder_stages.append(decoder_stage)
        
        # Output convolution
        self.out_conv = nn.Conv2d(features_per_stage[0], num_classes, 1)
        
        # Deep supervision outputs
        if deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList()
            for stage_idx in range(n_stages - 1):
                ds_ch = features_per_stage[-(stage_idx+2)]
                self.deep_supervision_outputs.append(
                    nn.Conv2d(ds_ch, num_classes, 1)
                )
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Encoder forward
        skip_features = []
        
        for stage in self.encoder_stages:
            x, skip_feature = stage(x)
            skip_features.append(skip_feature)
        
        # Decoder forward
        seg_outputs = []  # Deep supervision용
        
        for stage_idx, decoder_stage in enumerate(self.decoder_stages):
            # Skip feature (역순으로)
            skip_idx = len(skip_features) - 2 - stage_idx
            skip_feature = skip_features[skip_idx]
            
            # Decoder stage 실행
            x = decoder_stage(x, skip_feature)
            
            # Deep supervision output
            if self.deep_supervision and stage_idx < len(self.deep_supervision_outputs):
                seg_output = self.deep_supervision_outputs[stage_idx](x)
                seg_outputs.append(seg_output)
        
        # Final output
        final_output = self.out_conv(x)
        
        if self.deep_supervision:
            seg_outputs.append(final_output)
            return seg_outputs[::-1]  # 깊은 것부터 얕은 것 순서
        else:
            return final_output
    
    def compute_conv_feature_map_size(self, input_size):
        """Feature map 크기 계산"""
        total_size = 0
        current_size = input_size
        
        # Encoder
        for stage_idx in range(self.n_stages):
            stage_size = 1
            for dim in current_size:
                stage_size *= dim
            stage_size *= self.features_per_stage[stage_idx]
            total_size += stage_size
            
            if stage_idx < self.n_stages - 1:
                current_size = [dim // 2 for dim in current_size]
        
        # Decoder
        total_size *= 2
        
        return total_size
    
    @staticmethod
    def initialize(module):
        """가중치 초기화"""
        InitWeights_He(1e-2)(module)
    
    def print_model_info(self):
        """모델 정보 출력"""
        print("=== DSConvUNet Information ===")
        print(f"Input channels: {self.input_channels}")
        print(f"Number of stages: {self.n_stages}")
        print(f"Features per stage: {self.features_per_stage}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Deep supervision: {self.deep_supervision}")
        print(f"DSConv kernel size: {self.dsconv_kernel_size}")
        
        # Encoder stages 정보
        print("\n=== Encoder Stages ===")
        for i, stage in enumerate(self.encoder_stages):
            use_dsconv = (1 <= i <= min(self.n_stages-2, 3))
            has_pooling = (i < self.n_stages - 1)
            print(f"Stage {i}: {self.features_per_stage[i]} channels, "
                  f"DSConv: {use_dsconv}, Pooling: {has_pooling}")
        
        # Decoder stages 정보
        print("\n=== Decoder Stages ===")
        for i, stage in enumerate(self.decoder_stages):
            print(f"Stage {i}: DSConv enabled")


# 테스트 코드
if __name__ == '__main__':
    print("Testing Clean DSConvUNet...")
    
    model = DSConvUNet(
        input_channels=4,
        n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv2d,
        kernel_sizes=3,
        strides=(1, 2, 2, 2, 2, 2),
        n_conv_per_stage=2,
        num_classes=3,
        n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
        deep_supervision=True,
        dsconv_kernel_size=9
    ).cuda()
    
    print(model)
    model.print_model_info()

    
    # Forward pass 테스트
    data = torch.rand((2, 4, 256, 256)).cuda()
    with torch.no_grad():
        output = model(data)
        if isinstance(output, (list, tuple)):
            print(f"\n✅ Output shapes (deep supervision): {[x.shape for x in output]}")
        else:
            print(f"\n✅ Output shape: {output.shape}")
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("✅ Clean DSConvUNet test completed!")
