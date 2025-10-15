from typing import Union, Tuple, List
from dynamic_network_architectures.building_blocks.helper import get_matching_batchnorm
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.net.DSConvUNet import DSConvUNet
import pydoc

class nnUNetTrainerDSC(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   allow_init=True,
                                   dsconv_kernel_size: int = 9,
                                   extend_scope: float = 1.0,
                                   if_offset: bool = True
                                ) -> nn.Module:
        
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        if enable_deep_supervision is not None:
            arch_init_kwargs['deep_supervision'] = enable_deep_supervision
        arch_init_kwargs['dsconv_kernel_size'] = dsconv_kernel_size
        arch_init_kwargs['extend_scope'] = extend_scope
        arch_init_kwargs['if_offset'] = if_offset

        network = DSConvUNet(
                          input_channels=num_input_channels,
                          num_classes=num_output_channels,
                          **arch_init_kwargs
                          )
        if hasattr(network, 'initialize') and allow_init:
            network.apply(network.initialize)

        return network

