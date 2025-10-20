# nnUNetTrainerDSCResidual.py (최종 수정 버전)
from typing import Union, Tuple, List
import torch
from torch import nn
import pydoc

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# DSC UNet 모델이 정의된 파일 경로를 정확하게 지정해야 합니다.
# 예: from nnunetv2.training.net.my_custom_unet import CompleteDSCResidualUNet
from nnunetv2.training.net.DSConvUNet import DSCUNet


class nnUNetTrainerDSCResidual(nnUNetTrainer):
    """
    nnU-Net 프레임워크와 호환되도록 수정된 Dynamic Snake Convolution Residual UNet 트레이너.
    - 파라미터 처리를 간소화하고 nnU-Net의 plans.json 설정을 존중합니다.
    - Deep Supervision 설정을 동적으로 받아옵니다.
    """

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        nnU-Net의 plans.json 파일로부터 받은 파라미터를 사용하여
        CompleteDSCResidualUNet 네트워크를 빌드합니다.
        """
        print(f"[{__class__.__name__}] 🚀 Building CompleteDSCResidualUNet...")
        
        # pydoc.locate를 사용하여 문자열로 된 클래스(conv_op, norm_op 등)를 실제 클래스로 변환합니다.
        for ri in arch_init_kwargs_req_import:
            if arch_init_kwargs.get(ri) is not None:
                arch_init_kwargs[ri] = pydoc.locate(arch_init_kwargs[ri])

        # nnU-Net plans에서 제공하는 파라미터를 가져옵니다.
        # nnU-Net이 이미 적절한 형태로 가공해서 주므로, 복잡한 전처리가 필요 없습니다.
        features_per_stage = arch_init_kwargs['features_per_stage']
        n_stages = len(features_per_stage)
        
        # 디버깅을 위한 파라미터 출력
        print(f"[{__class__.__name__}] ─ Network Parameters ─")
        print(f" ┃ Input Channels: {num_input_channels}, Output Channels: {num_output_channels}")
        print(f" ┃ Stages: {n_stages}")
        print(f" ┃ Features per Stage: {features_per_stage}")
        print(f" ┃ Conv Op: {arch_init_kwargs['conv_op'].__name__}")
        print(f" ┃ Kernel Sizes: {arch_init_kwargs['kernel_sizes']}")
        print(f" ┃ Strides: {arch_init_kwargs['strides']}")
        print(f" ┃ Norm Op: {arch_init_kwargs['norm_op'].__name__}")
        print(f" ┃ Deep Supervision Enabled: {enable_deep_supervision}")
        print(f"────────────────────────────")

        # 네트워크 생성
        # arch_init_kwargs에 있는 대부분의 파라미터를 그대로 전달합니다.
        network = DSCUNet(
            input_channels=num_input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=arch_init_kwargs['conv_op'],
            kernel_sizes=arch_init_kwargs['kernel_sizes'],
            strides=arch_init_kwargs['strides'],
            n_conv_per_stage=arch_init_kwargs['n_conv_per_stage'], # Residual UNet은 n_blocks_per_stage를 사용
            num_classes=num_output_channels,
            n_conv_per_stage_decoder=arch_init_kwargs['n_conv_per_stage_decoder'],
            conv_bias=arch_init_kwargs['conv_bias'],
            norm_op=arch_init_kwargs['norm_op'],
            norm_op_kwargs=arch_init_kwargs['norm_op_kwargs'],
            dropout_op=arch_init_kwargs.get('dropout_op', None),
            dropout_op_kwargs=arch_init_kwargs.get('dropout_op_kwargs', None),
            nonlin=arch_init_kwargs['nonlin'],
            nonlin_kwargs=arch_init_kwargs['nonlin_kwargs'],
            deep_supervision=enable_deep_supervision,
        )

        print(f"[{__class__.__name__}] ✓ CompleteDSCResidualUNet created successfully.")

        # 가중치 초기화 (모델에 'initialize' 메소드가 있는 경우)
        if hasattr(network, 'initialize'):
            network.initialize(network)
            print(f"[{__class__.__name__}] ✓ Network weights initialized.")

        return network
