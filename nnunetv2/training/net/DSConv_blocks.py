# nnunetv2/nets/DSConv_blocks.py
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple
import numpy as np

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list, get_matching_pool_op
)
from dynamic_network_architectures.building_blocks.convnormnonlin import ConvDropoutNormNonlin
from .S3_DSConv_pro import DSConv_pro


class DSConvMultiViewBlock(nn.Module):
    """
    DSCNet의 multi-view convolution block을 nnU-Net 스타일로 구현
    일반 conv + x축 DSConv + y축 DSConv를 병렬로 실행 후 fusion
    """
    def __init__(self, input_channels, output_channels, kernel_size=9, 
                 extend_scope=1.0, if_offset=True, device='cuda',
                 norm_op=nn.BatchNorm2d, dropout_op=None, nonlin=nn.ReLU):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # 각 branch의 출력 채널 수
        branch_channels = output_channels
        
        # 1) 일반 convolution branch
        self.conv_standard = nn.Sequential(
            nn.Conv2d(input_channels, branch_channels, 3, padding=1),
            norm_op(branch_channels) if norm_op else nn.Identity(),
            nonlin(inplace=True) if nonlin else nn.Identity()
        )
        
        # 2) x축 방향 DSConv branch
        self.conv_x = DSConv_pro(
            input_channels, branch_channels, kernel_size, extend_scope, 
            morph=0, if_offset=if_offset, device=device
        )
        
        # 3) y축 방향 DSConv branch  
        self.conv_y = DSConv_pro(
            input_channels, branch_channels, kernel_size, extend_scope,
            morph=1, if_offset=if_offset, device=device
        )
        
        # 4) Fusion layer (3*branch_channels -> output_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * branch_channels, output_channels, 1),
            norm_op(output_channels) if norm_op else nn.Identity(),
            nonlin(inplace=True) if nonlin else nn.Identity()
        )
        
        # 5) Dropout if specified
        self.dropout = dropout_op(0.5) if dropout_op else None
        
    def forward(self, x):
        # 3개 branch 병렬 실행
        out_standard = self.conv_standard(x)
        out_x = self.conv_x(x)  
        out_y = self.conv_y(x)
        
        # Concatenate and fusion
        concatenated = torch.cat([out_standard, out_x, out_y], dim=1)
        fused = self.fusion(concatenated)
        
        if self.dropout is not None:
            fused = self.dropout(fused)
            
        return fused


class DSConvBlock(nn.Module):
    """
    단순한 DSConv block (x축 또는 y축 하나만)
    """
    def __init__(self, input_channels, output_channels, kernel_size=9,
                 extend_scope=1.0, if_offset=True, morph=0, device='cuda',
                 norm_op=nn.BatchNorm2d, dropout_op=None, nonlin=nn.ReLU):
        super().__init__()
        
        self.dsconv = DSConv_pro(
            input_channels, output_channels, kernel_size, extend_scope,
            morph=morph, if_offset=if_offset, device=device
        )
        
        self.norm = norm_op(output_channels) if norm_op else nn.Identity()
        self.nonlin = nonlin(inplace=True) if nonlin else nn.Identity()
        self.dropout = dropout_op(0.5) if dropout_op else None
        
    def forward(self, x):
        x = self.dsconv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x


class DSConvEncoder(PlainConvEncoder):
    """
    DSConv가 통합된 encoder
    PlainConvEncoder를 상속받아 일부 stage에서 DSConv 사용
    """
    def __init__(self, input_channels: int, n_stages: int, features_per_stage, conv_op: Type[_ConvNd],
                 kernel_sizes, strides, n_conv_per_stage, conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None, norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None, dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None, nonlin_kwargs: dict = None,
                 return_skips: bool = False, nonlin_first: bool = False,
                 # DSConv specific parameters
                 dsconv_kernel_size: int = 9, extend_scope: float = 1.0, if_offset: bool = True,
                 use_dsconv_stages: List[bool] = None):
        
        # DSConv 파라미터 저장
        self.dsconv_kernel_size = dsconv_kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.use_dsconv_stages = use_dsconv_stages or [False] * n_stages
        
        # 부모 클래스 초기화
        super().__init__(input_channels, n_stages, features_per_stage, conv_op,
                        kernel_sizes, strides, n_conv_per_stage, conv_bias,
                        norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, return_skips, nonlin_first)
        
        # DSConv stages 교체
        self._replace_dsconv_stages()
    
    def _replace_dsconv_stages(self):
        """
        지정된 stage들을 DSConv block으로 교체
        """
        for stage_idx, use_dsconv in enumerate(self.use_dsconv_stages):
            if use_dsconv and stage_idx < len(self.stages):
                # 해당 stage의 첫 번째 conv를 DSConv로 교체
                stage = self.stages[stage_idx]
                if len(stage) > 1 and hasattr(stage[1], 'conv'):  # stage[0]은 downsample, stage[1]이 conv
                    first_conv_block = stage[1]
                    if hasattr(first_conv_block, 'conv'):
                        # 기존 conv의 파라미터 가져오기
                        in_channels = first_conv_block.conv.in_channels
                        out_channels = first_conv_block.conv.out_channels
                        
                        # DSConv multi-view block으로 교체
                        dsconv_block = DSConvMultiViewBlock(
                            input_channels=in_channels,
                            output_channels=out_channels,
                            kernel_size=self.dsconv_kernel_size,
                            extend_scope=self.extend_scope,
                            if_offset=self.if_offset,
                            device='cuda',
                            norm_op=self.norm_op,
                            dropout_op=self.dropout_op,
                            nonlin=self.nonlin
                        )
                        
                        # 첫 번째 블록의 conv 교체
                        first_conv_block.conv = dsconv_block


class DSConvDecoderBlock(nn.Module):
    """
    DSConv Decoder Block - 3-branch 구조 또는 lightweight 버전
    """
    def __init__(self, input_channels, output_channels, kernel_size=7,
                 extend_scope=0.8, if_offset=True, device='cuda',
                 norm_op=nn.BatchNorm2d, dropout_op=None, nonlin=nn.ReLU,
                 use_lightweight=True):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_lightweight = use_lightweight
        
        if use_lightweight:
            # 가벼운 버전: 단일 DSConv만 사용
            self.dsconv = DSConv_pro(
                input_channels, output_channels, kernel_size, extend_scope,
                morph=0, if_offset=if_offset, device=device
            )
            
            self.norm = norm_op(output_channels) if norm_op else nn.Identity()
            self.nonlin = nonlin(inplace=True) if nonlin else nn.Identity()
            self.dropout = dropout_op(0.2) if dropout_op else None
        else:
            # 전체 3-branch 버전
            mid_channels = max(output_channels // 2, 16)
            
            # 1) 일반 convolution branch
            self.conv_standard = nn.Sequential(
                nn.Conv2d(input_channels, mid_channels, 3, padding=1),
                norm_op(mid_channels) if norm_op else nn.Identity(),
                nonlin(inplace=True) if nonlin else nn.Identity()
            )
            
            # 2) x축 방향 DSConv branch
            self.conv_x = DSConv_pro(
                input_channels, mid_channels, kernel_size, extend_scope,
                morph=0, if_offset=if_offset, device=device
            )
            
            # 3) y축 방향 DSConv branch
            self.conv_y = DSConv_pro(
                input_channels, mid_channels, kernel_size, extend_scope,
                morph=1, if_offset=if_offset, device=device
            )
            
            # 4) Fusion layer
            self.fusion = nn.Sequential(
                nn.Conv2d(3 * mid_channels, output_channels, 1),
                norm_op(output_channels) if norm_op else nn.Identity(),
                nonlin(inplace=True) if nonlin else nn.Identity()
            )
            
            self.dropout = dropout_op(0.2) if dropout_op else None
    
    def forward(self, x):
        if self.use_lightweight:
            # 가벼운 버전
            out = self.dsconv(x)
            out = self.norm(out)
            out = self.nonlin(out)
            if self.dropout is not None:
                out = self.dropout(out)
            return out
        else:
            # 전체 3-branch 버전
            out_standard = self.conv_standard(x)
            out_x = self.conv_x(x)
            out_y = self.conv_y(x)
            
            concatenated = torch.cat([out_standard, out_x, out_y], dim=1)
            fused = self.fusion(concatenated)
            
            if self.dropout is not None:
                fused = self.dropout(fused)
            
            return fused


class DSConvDecoder(UNetDecoder):
    """
    DSConv가 통합된 Decoder
    UNetDecoder를 상속받아 선택적으로 DSConv block 사용
    """
    def __init__(self, encoder, num_classes: int, n_conv_per_stage_decoder,
                 deep_supervision: bool, nonlin_first: bool = False,
                 use_dsconv_in_decoder: List[bool] = None,
                 dsconv_kernel_size: int = 7,
                 dsconv_extend_scope: float = 0.8,
                 dsconv_if_offset: bool = True,
                 use_lightweight_dsconv: bool = True):
        
        # DSConv 파라미터 저장
        self.use_dsconv_in_decoder = use_dsconv_in_decoder or [False] * len(n_conv_per_stage_decoder)
        self.dsconv_kernel_size = dsconv_kernel_size
        self.dsconv_extend_scope = dsconv_extend_scope
        self.dsconv_if_offset = dsconv_if_offset
        self.use_lightweight_dsconv = use_lightweight_dsconv
        
        # 부모 클래스 초기화
        super().__init__(encoder, num_classes, n_conv_per_stage_decoder,
                        deep_supervision, nonlin_first)
        
        # DSConv stages 구성
        self._build_dsconv_stages()
    
    def _build_dsconv_stages(self):
        """지정된 decoder stage를 DSConv로 교체"""
        for stage_idx, use_dsconv in enumerate(self.use_dsconv_in_decoder):
            if use_dsconv and stage_idx < len(self.stages):
                self._replace_stage_with_dsconv(stage_idx)
    
    def _replace_stage_with_dsconv(self, stage_idx):
        """특정 stage를 DSConv로 교체"""
        original_stage = self.stages[stage_idx]
        
        # 기존 stage에서 정보 추출
        if len(original_stage) >= 2:
            # Transpose convolution (upsampling) 유지
            transpose_conv = original_stage[0]
            
            # 첫 번째 convolution block에서 채널 정보 추출
            conv_block = original_stage[1]
            if hasattr(conv_block, 'conv'):
                input_features = conv_block.conv.in_channels
                output_features = conv_block.conv.out_channels
            else:
                # Fallback: 추정
                input_features = 256  # 기본값
                output_features = 128
            
            # DSConv block 생성
            dsconv_block = DSConvDecoderBlock(
                input_channels=input_features,
                output_channels=output_features,
                kernel_size=self.dsconv_kernel_size,
                extend_scope=self.dsconv_extend_scope,
                if_offset=self.dsconv_if_offset,
                device='cuda',
                norm_op=self.encoder.norm_op,
                dropout_op=self.encoder.dropout_op,
                nonlin=self.encoder.nonlin,
                use_lightweight=self.use_lightweight_dsconv
            )
            
            # 새로운 stage 구성
            new_stage_components = [transpose_conv, dsconv_block]
            
            # 나머지 conv blocks 추가 (필요한 경우)
            for i in range(2, len(original_stage)):
                new_stage_components.append(original_stage[i])
            
            # 새로운 stage로 교체
            self.stages[stage_idx] = nn.Sequential(*new_stage_components)
