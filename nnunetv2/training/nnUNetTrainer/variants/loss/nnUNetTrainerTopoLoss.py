import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss,DC_and_SC_loss,DC_and_CE_SCloss,DC_and_BettiMatchingLoss,DC_and_WassersteinLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerDCBettiLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_and_BettiMatchingLoss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            relative=True,
            filtration='superlevel',
            weight_dice=1.0,
            weight_topo=1.0,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        # if self._do_i_compile():
        #     loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp:
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

class nnUNetTrainerDCWassersteinLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_and_WassersteinLoss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            relative=False,
            filtration='superlevel',
            dimensions=[0, 1],
            weight_dice=1.0,
            weight_topo=1.0,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        # if self._do_i_compile():
        #     loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp :
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
