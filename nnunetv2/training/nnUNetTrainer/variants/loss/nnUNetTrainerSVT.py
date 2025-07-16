import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_SoftGradientDiffTVLoss_loss,DC_CE_SoftGradientDiffTVLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerCole import nnUNetTrainerCole 

class nnUNetTrainerSVT(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_SoftGradientDiffTVLoss_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            weight_dice=1.0,
            weight_length=1.0,
            weight_tv=1.0,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

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

class nnUNetTrainerSVTCE(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_CE_SoftGradientDiffTVLoss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            weight_length=1.0,
            weight_tv=1.0,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

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


class nnUNetTrainerSVTCole(nnUNetTrainerCole):
    def _build_loss(self):
        loss = DC_SoftGradientDiffTVLoss_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            weight_dice=1.0,
            weight_length=1.0,
            weight_tv=1.0,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

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

class nnUNetTrainerSVTCECole(nnUNetTrainerCole):
    def _build_loss(self):
        loss = DC_CE_SoftGradientDiffTVLoss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            ce_kwargs={},
            weight_ce=1.0,
            weight_dice=1.0,
            weight_length=1.0,
            weight_tv=1.0,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

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

