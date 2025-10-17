# nnUNetTrainerDSCResidual.py (수정된 버전)
"""
Fixed DSC Residual UNet Trainer for nnU-Net
Handles parameter length mismatch issues
"""

from typing import Union, Tuple, List
import torch
from torch import nn
import pydoc

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.net.DSConvUNet import CompleteDSCResidualUNet


class nnUNetTrainerDSCResidual(nnUNetTrainer):
    """
    DSC Residual UNet Trainer - Fixed kernel_sizes handling
    """
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build DSC Residual UNet - Fixed kernel_sizes handling
        """
        print(f"[DSC Trainer] Building DSC Residual UNet...")
        print(f"[DSC Trainer] Input channels: {num_input_channels}, Output channels: {num_output_channels}")
        print(f"[DSC Trainer] Deep supervision: {enable_deep_supervision}")
        
        # Process required imports
        for ri in arch_init_kwargs_req_import:
            if arch_init_kwargs[ri] is not None:
                arch_init_kwargs[ri] = pydoc.locate(arch_init_kwargs[ri])
        
        # Debug: Print raw parameters
        print(f"[DSC Trainer] Raw arch_init_kwargs:")
        for key, value in arch_init_kwargs.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        # Extract parameters
        conv_op = arch_init_kwargs['conv_op']
        norm_op = arch_init_kwargs['norm_op']
        
        # === FIX: Handle kernel_sizes properly ===
        raw_kernel_sizes = arch_init_kwargs['kernel_sizes']
        print(f"[DSC Trainer] Raw kernel_sizes: {raw_kernel_sizes} (type: {type(raw_kernel_sizes)})")
        
        # Convert kernel_sizes to proper format
        if isinstance(raw_kernel_sizes, list):
            processed_kernel_sizes = []
            for ks in raw_kernel_sizes:
                if isinstance(ks, (list, tuple)):
                    # For 3D: (3,3,3) -> 3 (take first element)
                    processed_kernel_sizes.append(ks[0])
                else:
                    # For 2D: 3 -> 3 (keep as is)
                    processed_kernel_sizes.append(ks)
        else:
            # Single value
            if isinstance(raw_kernel_sizes, (list, tuple)):
                processed_kernel_sizes = [raw_kernel_sizes[0]] * 5  # Default 5 stages
            else:
                processed_kernel_sizes = [raw_kernel_sizes] * 5
        
        print(f"[DSC Trainer] Processed kernel_sizes: {processed_kernel_sizes}")
        
        # === Get other parameters ===
        features_per_stage = arch_init_kwargs['features_per_stage']
        strides_raw = arch_init_kwargs['strides']
        n_blocks_per_stage = arch_init_kwargs['n_conv_per_stage']
        n_conv_per_stage_decoder = arch_init_kwargs['n_conv_per_stage_decoder']
        
        # Process strides
        dsc_strides = [1]  # First stage
        for stride in strides_raw:
            if isinstance(stride, (list, tuple)):
                dsc_strides.append(stride[0])
            else:
                dsc_strides.append(stride)
        
        n_stages = len(features_per_stage)
        
        print(f"[DSC Trainer] Final parameters:")
        print(f"  n_stages: {n_stages}")
        print(f"  features_per_stage: {features_per_stage}")
        print(f"  processed_kernel_sizes: {processed_kernel_sizes}")
        print(f"  strides: {dsc_strides}")
        
        # DSC parameters
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dsc_params = {
            'extend_scope': 1.0,
            'if_offset': True,
            'device': device
        }
        
        # Create network
        network = CompleteDSCResidualUNet(
            input_channels=num_input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=processed_kernel_sizes,  # Use processed kernel sizes
            strides=dsc_strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_output_channels,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=arch_init_kwargs['conv_bias'],
            norm_op=norm_op,
            norm_op_kwargs=arch_init_kwargs['norm_op_kwargs'],
            dropout_op=arch_init_kwargs.get('dropout_op', None),
            dropout_op_kwargs=arch_init_kwargs.get('dropout_op_kwargs', {}),
            nonlin=arch_init_kwargs['nonlin'],
            nonlin_kwargs=arch_init_kwargs['nonlin_kwargs'],
            deep_supervision=enable_deep_supervision,
            **dsc_params
        )
        
        print(f"[DSC Trainer] ✓ DSC Residual UNet created successfully")
        
        # Initialize
        if hasattr(network, 'initialize'):
            network.initialize(network)
            print(f"[DSC Trainer] ✓ Network initialized")
        
        return network


# ========================================
# Alternative: Safer Version with More Debugging
# ========================================

# class nnUNetTrainerDSCResidualSafe(nnUNetTrainer):
#     """
#     Ultra-safe version with extensive debugging
#     """
    
#     @staticmethod
#     def build_network_architecture(architecture_class_name: str,
#                                    arch_init_kwargs: dict,
#                                    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
#                                    num_input_channels: int,
#                                    num_output_channels: int,
#                                    enable_deep_supervision: bool = True) -> nn.Module:
        
#         print(f"[Safe DSC Trainer] Starting network building...")
#         print(f"[Safe DSC Trainer] Architecture args keys: {list(arch_init_kwargs.keys())}")
        
#         # Debug: Print all arch_init_kwargs
#         for key, value in arch_init_kwargs.items():
#             print(f"[Safe DSC Trainer] {key}: {value} (type: {type(value)})")
        
#         try:
#             # Try DSC Residual UNet with safe parameters
#             conv_op = arch_init_kwargs.get('conv_op', nn.Conv2d)
            
#             # Use conservative default parameters
#             network = CompleteDSCResidualUNet(
#                 input_channels=num_input_channels,
#                 n_stages=5,  # Standard nnU-Net stages
#                 features_per_stage=(32, 64, 128, 256, 512),  # Standard progression
#                 conv_op=conv_op,
#                 kernel_sizes=3,  # Single kernel size
#                 strides=(1, 2, 2, 2, 2),  # Standard strides
#                 n_blocks_per_stage=2,  # Standard blocks
#                 num_classes=num_output_channels,
#                 n_conv_per_stage_decoder=(2, 2, 2, 2),  # Standard decoder
#                 conv_bias=False,
#                 norm_op=arch_init_kwargs.get('norm_op', nn.BatchNorm2d),
#                 norm_op_kwargs=arch_init_kwargs.get('norm_op_kwargs', {}),
#                 nonlin=arch_init_kwargs.get('nonlin', nn.ReLU),
#                 nonlin_kwargs={'inplace': True},
#                 deep_supervision=enable_deep_supervision,
#                 extend_scope=1.0,
#                 if_offset=True,
#                 device='cuda' if torch.cuda.is_available() else 'cpu'
#             )
            
#             print(f"[Safe DSC Trainer] ✓ DSC UNet created with safe parameters")
            
#             # Initialize
#             if hasattr(network, 'initialize'):
#                 network.initialize(network)
            
#             return network
            
#         except Exception as e:
#             print(f"[Safe DSC Trainer] ❌ DSC UNet failed: {e}")
#             print(f"[Safe DSC Trainer] Using standard nnU-Net...")
            
#             # Fallback to original nnU-Net
#             from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
#             return get_network_from_plans(
#                 arch_init_kwargs, arch_init_kwargs_req_import,
#                 num_input_channels, num_output_channels, allow_init=True,
#                 deep_supervision=enable_deep_supervision
#             )


# ========================================
# Usage
# ========================================

# if __name__ == "__main__":
#     from batchgenerators.utilities.file_and_folder_operations import load_json
    
#     # Your code with the fixed trainer
#     plans_file = "/data/image/project/ng_tube/nnunet/data/nnUNet_preprocessed/Dataset3008_NGT_hospitals_updated_25_09_30/nnUNetSegmentation.json"
#     config = '2d'
#     fold = 0
#     dataset_json_file = "/data/image/project/ng_tube/nnunet/data/nnUNet_preprocessed/Dataset3008_NGT_hospitals_updated_25_09_30/dataset.json"
    
#     plans = load_json(plans_file)
#     dataset_json = load_json(dataset_json_file)
    
#     # Use the safe version first
#     trainer = nnUNetTrainerDSCResidualSafe(plans, config, fold, dataset_json)
#     trainer.initialize(True)
#     trainer.run_training()
