from nnunetv2.training.loss.clce import dice_cldice_loss, CE_cldice_loss, CE_clCE_loss, dice_clCE_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch



class nnUNetTrainerDiceclCELoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
    def _build_loss(self):
        loss = dice_clCE_loss(iter_=3, smooth=1.0, weight_dice=1, weight_clCE=1)
        return loss


class nnUNetTrainerCEclCEloss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.enable_deep_supervision = False
    def _build_loss(self):
        loss = CE_clCE_loss({}, iter_=3, weight_ce=1, weight_clCE=1)
        return loss


class nnUNetTrainerDiceclCELossCole(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
    def _build_loss(self):
        loss = dice_clCE_loss(iter_=3, smooth=1.0, weight_dice=1, weight_clCE=1)
        return loss


class nnUNetTrainerCEclCElossCole(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.enable_deep_supervision = False
    def _build_loss(self):
        loss = CE_clCE_loss({}, iter_=3, weight_ce=1, weight_clCE=1)
        return loss