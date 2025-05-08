import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss,DC_and_SC_loss,DC_and_CE_SCloss, CE_Clloss, DC_Clloss, DC_and_CE_Clloss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.scloss import MultiClassOneVsRestSCLoss


class nnUNetTrainerDCCLLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_Clloss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            cl_kwargs={
                'iter_': 2,
                'smooth': 1.0,
                'exclude_background': True,
            },
            weight_dice=1.0,
            weight_cl=1.0,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

class nnUNetTrainerDCCECLLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_and_CE_Clloss({'batch_dice': self.configuration_manager.batch_dice,
                                'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
                                {},
                                {'iter_': 2,'smooth': 1.0,'exclude_background': True,},
                                weight_ce=1, weight_dice=1, weight_cl=1, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

class nnUNetTrainerCECLLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = CE_Clloss({},{'iter_': 2,'smooth': 1.0,'exclude_background': True,},weight_ce=1,weight_cl=1,ignore_label=self.label_manager.ignore_label,)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
