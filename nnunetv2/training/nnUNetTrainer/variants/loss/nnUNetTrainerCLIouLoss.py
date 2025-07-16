import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_ClIoULoss, DC_and_CE_ClIoULoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerCole import nnUNetTrainerCole

# --- Trainer with Dice + clIoU (no CE) ---
class nnUNetTrainerDCCLIoULoss(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_and_ClIoULoss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            clioU_kwargs={  # 사용되는 clIoU 인자
                'num_iter': 10,
                'eps': 1e-6
            },
            weight_dice=1.0,
            weight_cl=1.0,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

# --- Trainer with Dice + CE + clIoU ---
class nnUNetTrainerDCCECLIouLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = DC_and_CE_ClIoULoss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            clioU_kwargs={
                'num_iter': 10,
                'eps': 1e-6
            },
            ce_kwargs={},  # CE 활성화
            weight_ce=1.0,
            weight_dice=1.0,
            weight_cl=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

# --- Cole variant, Dice + clIoU ---
class nnUNetTrainerDCCLIoULossCole(nnUNetTrainerCole):
    def _build_loss(self):
        loss = DC_and_ClIoULoss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            clioU_kwargs={
                'num_iter': 10,
                'eps': 1e-6
            },
            weight_dice=1.0,
            weight_cl=1.0,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

# --- Cole variant, Dice + CE + clIoU ---
class nnUNetTrainerDCCECLIouLossCole(nnUNetTrainerCole):
    def _build_loss(self):
        loss = DC_and_CE_ClIoULoss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            clioU_kwargs={
                'num_iter': 10,
                'eps': 1e-6
            },
            ce_kwargs={},  # CE 활성화
            weight_ce=1.0,
            weight_dice=1.0,
            weight_cl=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
