# nnunetv2/nets/DSConvUNet.py - 완전 버전
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

# 절대 경로로 수정 (상대 경로 문제 해결)
try:
    from .S3_DSConv_pro import DSConv_pro
except ImportError:
    # 절대 경로로 fallback
    from nnunetv2.training.net.S3_DSConv_pro import DSConv_pro


class DSConvBlock(nn.Module):
    """
    메모리 최적화된 DSConv Multi-View Block
    일반 conv + x축 DSConv + y축 DSConv → fusion
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, 
                 extend_scope=1.0, if_offset=True, device='cuda',
                 use_checkpoint=True):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
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
    
    def _forward_standard_branch(self, x):
        """표준 convolution branch (checkpoint용)"""
        return self.relu(self.gn_standard(self.conv_standard(x)))
    
    def _forward_dsconv_x_branch(self, x):
        """x축 DSConv branch (checkpoint용)"""
        return self.conv_x(x)
    
    def _forward_dsconv_y_branch(self, x):
        """y축 DSConv branch (checkpoint용)"""
        return self.conv_y(x)
    
    def _forward_fusion(self, concatenated):
        """Fusion layer (checkpoint용)"""
        return self.relu(self.fusion_gn(self.fusion_conv(concatenated)))
        
    def forward(self, x):
        # 입력 텐서를 연속적으로 만들어 cuDNN 호환성 확보
        x = x.contiguous()
        
        if self.use_checkpoint and x.requires_grad:
            # Gradient checkpointing으로 각 branch 처리
            out_standard = checkpoint.checkpoint(self._forward_standard_branch, x, use_reentrant=False)
            out_x = checkpoint.checkpoint(self._forward_dsconv_x_branch, x, use_reentrant=False)
            out_y = checkpoint.checkpoint(self._forward_dsconv_y_branch, x, use_reentrant=False)
        else:
            # 일반 처리
            out_standard = self._forward_standard_branch(x)
            out_x = self._forward_dsconv_x_branch(x)
            out_y = self._forward_dsconv_y_branch(x)
        
        # Concatenate and fusion - 각 출력도 contiguous 확보
        concatenated = torch.cat([
            out_standard.contiguous(), 
            out_x.contiguous(), 
            out_y.contiguous()
        ], dim=1)
        
        # Fusion도 checkpoint 적용
        if self.use_checkpoint and concatenated.requires_grad:
            fused = checkpoint.checkpoint(self._forward_fusion, concatenated, use_reentrant=False)
        else:
            fused = self._forward_fusion(concatenated)
        
        return fused.contiguous()


class StandardConvBlock(nn.Module):
    """
    메모리 최적화된 일반 convolution block (DSConv 없는 구간용)
    """
    def __init__(self, in_channels, out_channels, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def _forward_impl(self, x):
        """실제 forward 구현 (checkpoint용)"""
        return self.relu(self.gn(self.conv(x)))
        
    def forward(self, x):
        x = x.contiguous()
        
        if self.use_checkpoint and x.requires_grad:
            result = checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            result = self._forward_impl(x)
            
        return result.contiguous()


class EncoderStage(nn.Module):
    """
    메모리 최적화된 Encoder Stage = Conv blocks + Pooling (마지막 stage 제외)
    """
    def __init__(self, in_channels, out_channels, n_conv_blocks, 
                 use_dsconv=False, dsconv_kernel_size=9, extend_scope=1.0, 
                 if_offset=True, has_pooling=True, use_checkpoint=True):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
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
                    device='cuda',
                    use_checkpoint=use_checkpoint
                )
            else:
                # 나머지는 일반 conv
                block = StandardConvBlock(
                    block_in_ch, out_channels,
                    use_checkpoint=use_checkpoint
                )
            
            self.conv_blocks.append(block)
        
        # Pooling (마지막 stage가 아닌 경우만)
        self.pool = nn.MaxPool2d(2) if has_pooling else None
    
    def _forward_conv_blocks(self, x):
        """Conv blocks 처리 (checkpoint용)"""
        for block in self.conv_blocks:
            x = block(x)
            x = x.contiguous()
        return x
    
    def forward(self, x):
        # 입력을 contiguous로 만들기
        x = x.contiguous()
        
        # Conv blocks 실행 (checkpoint 적용)
        if self.use_checkpoint and x.requires_grad and len(self.conv_blocks) > 1:
            x = checkpoint.checkpoint(self._forward_conv_blocks, x, use_reentrant=False)
        else:
            x = self._forward_conv_blocks(x)
        
        # Skip connection을 위해 pooling 전 feature 저장
        skip_feature = x.contiguous()
        
        # Pooling (있는 경우만)
        if self.pool is not None:
            x = self.pool(x).contiguous()
        
        return x, skip_feature


class DecoderStage(nn.Module):
    """
    메모리 최적화된 Decoder Stage = Upsample + Skip Connection + Conv blocks
    """
    def __init__(self, in_channels, skip_channels, out_channels, n_conv_blocks,
                 use_dsconv=True, dsconv_kernel_size=9, extend_scope=1.0, 
                 if_offset=True, use_checkpoint=True):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Conv blocks (skip connection 후)
        self.conv_blocks = nn.ModuleList()
        concat_channels = in_channels + skip_channels
        
        for block_idx in range(n_conv_blocks):
            block_in_ch = concat_channels if block_idx == 0 else out_channels
            
            if use_dsconv and block_idx == 0:
                # 첫 번째 블록은 DSConv
                block = DSConvBlock(
                    block_in_ch, out_channels,
                    kernel_size=dsconv_kernel_size,
                    extend_scope=extend_scope,
                    if_offset=if_offset,
                    device='cuda',
                    use_checkpoint=use_checkpoint
                )
            else:
                # 나머지는 일반 conv
                block = StandardConvBlock(
                    block_in_ch, out_channels,
                    use_checkpoint=use_checkpoint
                )
            
            self.conv_blocks.append(block)
    
    def _forward_conv_blocks(self, x):
        """Conv blocks 처리 (checkpoint용)"""
        for block in self.conv_blocks:
            x = block(x)
            x = x.contiguous()
        return x
    
    def forward(self, x, skip_feature):
        # Upsampling 후 contiguous 확보
        x = self.upsample(x).contiguous()
        
        # Skip connection - 양쪽 텐서 모두 contiguous 확보
        x = torch.cat([
            x.contiguous(), 
            skip_feature.contiguous()
        ], dim=1)
        
        # Conv blocks 실행 (checkpoint 적용)
        if self.use_checkpoint and x.requires_grad and len(self.conv_blocks) > 1:
            x = checkpoint.checkpoint(self._forward_conv_blocks, x, use_reentrant=False)
        else:
            x = self._forward_conv_blocks(x)
        
        return x


class nnUNetDecoder(nn.Module):
    """
    nnUNet 호환성을 위한 Decoder Wrapper 클래스
    """
    def __init__(self, decoder_stages, deep_supervision_outputs, num_classes, deep_supervision=True):
        super().__init__()
        self.stages = decoder_stages
        self.deep_supervision_outputs = deep_supervision_outputs
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
    def enable_deep_supervision(self):
        """Deep supervision 활성화"""
        self.deep_supervision = True
        
    def disable_deep_supervision(self):
        """Deep supervision 비활성화"""
        self.deep_supervision = False


class DSConvUNet(nn.Module):
    """
    Gradient Checkpointing이 적용된 메모리 최적화 DSConvUNet - nnUNet 완전 호환 버전
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
                 if_offset: bool = True,
                 # 메모리 최적화 parameters
                 use_gradient_checkpointing: bool = False,
                 checkpoint_segments: int = 2
                 ):
        super().__init__()
        
        # 메모리 최적화 설정
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpoint_segments = checkpoint_segments
        
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
        self.enable_deep_supervision = deep_supervision  # nnUNet 호환성
        
        # DSConv 설정
        self.dsconv_kernel_size = dsconv_kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        
        # Encoder stages 구성 (checkpoint 적용)
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
                has_pooling=has_pooling,
                use_checkpoint=self.use_gradient_checkpointing
            )
            
            self.encoder_stages.append(encoder_stage)
        
        # Encoder wrapper (nnUNet 호환성)
        self.encoder = nn.Module()
        self.encoder.stages = self.encoder_stages
        self.encoder.output_channels = features_per_stage
        
        # Decoder stages 구성 (checkpoint 적용)
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
                if_offset=if_offset,
                use_checkpoint=self.use_gradient_checkpointing
            )
            
            self.decoder_stages.append(decoder_stage)
        
        # Output convolution
        self.out_conv = nn.Conv2d(features_per_stage[0], num_classes, 1)
        
        # Deep supervision outputs
        self.deep_supervision_outputs = None
        if deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList()
            for stage_idx in range(n_stages - 1):
                ds_ch = features_per_stage[-(stage_idx+2)]
                self.deep_supervision_outputs.append(
                    nn.Conv2d(ds_ch, num_classes, 1)
                )
        
        # nnUNet 호환성을 위한 Decoder wrapper 생성
        self.decoder = nnUNetDecoder(
            decoder_stages=self.decoder_stages,
            deep_supervision_outputs=self.deep_supervision_outputs,
            num_classes=num_classes,
            deep_supervision=deep_supervision
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """nnUNet 호환성을 위한 deep supervision 제어"""
        self.deep_supervision = enabled
        self.enable_deep_supervision = enabled
        
        # Decoder의 deep supervision 속성 업데이트
        if hasattr(self, 'decoder'):
            self.decoder.deep_supervision = enabled
            
        # Deep supervision outputs 제어
        if self.deep_supervision_outputs is not None:
            for ds_output in self.deep_supervision_outputs:
                for param in ds_output.parameters():
                    param.requires_grad_(enabled)
    
    def enable_deep_supervision_training(self):
        """Deep supervision 활성화 (nnUNet 호환성)"""
        self.set_deep_supervision_enabled(True)
    
    def disable_deep_supervision_training(self):
        """Deep supervision 비활성화 (nnUNet 호환성)"""
        self.set_deep_supervision_enabled(False)
    
    def enable_checkpointing(self):
        """런타임에 checkpointing 활성화"""
        self.use_gradient_checkpointing = True
        for stage in self.encoder_stages + self.decoder_stages:
            if hasattr(stage, 'use_checkpoint'):
                stage.use_checkpoint = True
            for block in stage.conv_blocks:
                if hasattr(block, 'use_checkpoint'):
                    block.use_checkpoint = True
    
    def disable_checkpointing(self):
        """런타임에 checkpointing 비활성화 (추론 시)"""
        self.use_gradient_checkpointing = False
        for stage in self.encoder_stages + self.decoder_stages:
            if hasattr(stage, 'use_checkpoint'):
                stage.use_checkpoint = False
            for block in stage.conv_blocks:
                if hasattr(block, 'use_checkpoint'):
                    block.use_checkpoint = False
    
    def forward(self, x):
        # 학습 중에만 checkpointing 활성화
        if self.training and self.use_gradient_checkpointing:
            self.enable_checkpointing()
        else:
            self.disable_checkpointing()
        
        # 입력 텐서를 먼저 contiguous로 만들기
        x = x.contiguous()
        
        # Encoder forward
        skip_features = []
        
        for stage in self.encoder_stages:
            x, skip_feature = stage(x)
            # Skip feature도 contiguous 확보
            skip_features.append(skip_feature.contiguous())
        
        # Decoder forward
        seg_outputs = []  # Deep supervision용
        
        for stage_idx, decoder_stage in enumerate(self.decoder_stages):
            # Skip feature (역순으로)
            skip_idx = len(skip_features) - 2 - stage_idx
            skip_feature = skip_features[skip_idx]
            
            # Decoder stage 실행 - 입력도 contiguous 확보
            x = decoder_stage(x.contiguous(), skip_feature)
            
            # Deep supervision output
            if self.deep_supervision and self.deep_supervision_outputs and stage_idx < len(self.deep_supervision_outputs):
                seg_output = self.deep_supervision_outputs[stage_idx](x.contiguous())
                seg_outputs.append(seg_output)
        
        # Final output
        final_output = self.out_conv(x.contiguous())
        
        if self.deep_supervision and seg_outputs:
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
        """모델 정보 출력 (메모리 최적화 정보 포함)"""
        print("=== Memory-Optimized DSConvUNet Information ===")
        print(f"Input channels: {self.input_channels}")
        print(f"Number of stages: {self.n_stages}")
        print(f"Features per stage: {self.features_per_stage}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Deep supervision: {self.deep_supervision}")
        print(f"DSConv kernel size: {self.dsconv_kernel_size}")
        print(f"🔧 Gradient Checkpointing: {self.use_gradient_checkpointing}")
        print(f"🔧 Checkpoint Segments: {self.checkpoint_segments}")
        
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
            print(f"Stage {i}: DSConv enabled, checkpoint optimized")
        
        # nnUNet 호환성 확인
        print("\n=== nnUNet Compatibility ===")
        print(f"Has decoder attribute: {hasattr(self, 'decoder')}")
        print(f"Has encoder attribute: {hasattr(self, 'encoder')}")
        print(f"Decoder has deep_supervision: {hasattr(self.decoder, 'deep_supervision') if hasattr(self, 'decoder') else False}")
        print(f"Has set_deep_supervision_enabled method: {hasattr(self, 'set_deep_supervision_enabled')}")
        
        # 메모리 사용량 추정
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        
        if self.use_gradient_checkpointing:
            print("메모리 사용량 약 50-70% 절약 예상 (시간 10-20% 증가)")
        else:
            print("Full gradient storage (높은 메모리 사용량)")
        print("cuDNN contiguous memory compatibility added")


# 테스트 코드
if __name__ == '__main__':
    print("Testing Complete Memory-Optimized DSConvUNet...")
    
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
        dsconv_kernel_size=9,
        use_gradient_checkpointing=True  # 메모리 최적화 활성화
    ).cuda()
    
    model.print_model_info()
    
    # nnUNet 호환성 테스트
    print("\n=== nnUNet Compatibility Test ===")
    print(model)
    
    # Deep supervision 제어 테스트
    print("Testing deep supervision control...")
    model.set_deep_supervision_enabled(False)
    print(f"Deep supervision disabled: {model.decoder.deep_supervision}")
    
    model.set_deep_supervision_enabled(True)
    print(f"Deep supervision enabled: {model.decoder.deep_supervision}")
    
    # Forward pass 테스트
    print("\n=== 메모리 최적화 테스트 ===")
    data = torch.rand((1, 4, 640, 768)).cuda()  # 작은 크기로 테스트
    
    print("Gradient Checkpointing 활성화 상태로 forward pass 테스트...")
    try:
        model.train()  # 학습 모드
        with torch.cuda.amp.autocast():  # Mixed precision 함께 사용
            output = model(data)
            if isinstance(output, (list, tuple)):
                print(f"출력 형태 (deep supervision): {[x.shape for x in output]}")
            else:
                print(f"출력 형태: {output.shape}")
        print("메모리 최적화 성공!")
    except RuntimeError as e:
        print(f"오류: {e}")
    
    print("Complete Memory-Optimized DSConvUNet 준비 완료!")
