import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss, SoftSkeletonRecallLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.scloss import BinarySCLoss,MultiClassOneVsRestSCLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.cldice import ClDiceLoss
from torch import nn
from torch import Tensor
from topolosses.losses.hutopo import HutopoLoss
from topolosses.losses.betti_matching import BettiMatchingLoss
import torch.nn.functional as F
import os
from nnunetv2.training.loss.softgradienttv import SoftGradientDiffTVLoss
from nnunetv2.training.loss.cliou import ClIoULoss
from nnunetv2.training.loss.recall_panelty_loss import SpuriousBranchPenaltyLoss

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_SC_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs: dict,
        sc_kwargs: dict,
        weight_dice: float = 1.0,
        weight_sc: float = 1.0,
        dice_class=MemoryEfficientSoftDiceLoss
    ):
        """
        DC (Dice)와 SC (Spatial Coherence)를 합친 복합 손실함수

        Args:
            num_classes (int): 클래스 수
            soft_dice_kwargs (dict): Dice 손실 설정 파라미터
            sc_kwargs (dict): SC 손실 설정 파라미터 (k, alpha)
            weight_dice (float): Dice 손실 가중치
            weight_sc (float): SC 손실 가중치
            do_bg (bool): SC에서 배경(클래스 0)을 제외할지 여부
            dice_class: 사용할 Dice Loss 클래스
        """
        super().__init__()

        self.weight_dice = weight_dice
        self.weight_sc = weight_sc

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1,**soft_dice_kwargs)
        self.sc = MultiClassOneVsRestSCLoss(**sc_kwargs)

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            net_output (Tensor): [B, C, H, W] - 로짓 출력
            target (Tensor): [B, H, W] - 정답 클래스 인덱스

        Returns:
            loss (Tensor): 최종 손실값
        """
        sc_loss = self.sc(net_output, target)
        dice_loss = self.dc(net_output, target)

        return self.weight_dice * dice_loss + self.weight_sc * sc_loss
    

class DC_Clloss(nn.Module):
    def __init__(self, soft_dice_kwargs, cl_kwargs, weight_dice=1, weight_cl=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_Clloss, self).__init__()

        self.weight_dice = weight_dice
        self.weight_dice = weight_dice
        self.weight_cl = weight_cl
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cl = ClDiceLoss(**cl_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        cl_loss = self.cl(net_output,target)

        result =  self.weight_dice * dc_loss + self.weight_cl *cl_loss
        return result


class DC_and_BettiMatchingLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, weight_topo=1, weight_dice=1,dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_topo = weight_topo

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.topo = BettiMatchingLoss(softmax=True,use_base_loss=False,num_processes=os.cpu_count())

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        dc_loss = self.dc(net_output, target)

        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target_onehot = F.one_hot(target.long(), num_classes=net_output.shape[1]) #(B,H,W,C)
        target_onehot = target_onehot.permute(0, -1, *range(1, target.dim())).float()
        topo_loss = self.topo(net_output, target_onehot)
        
        return self.weight_dice * dc_loss + self.weight_topo * topo_loss


class DC_and_WassersteinLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, weight_topo=1, weight_dice=1,
                 dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_topo = weight_topo

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.topo = HutopoLoss(softmax=True,use_base_loss=False,num_processes=os.cpu_count())

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        dc_loss = self.dc(net_output, target)

        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target_onehot = F.one_hot(target.long(), num_classes=net_output.shape[1]) #(B,H,W,C)
        target_onehot = target_onehot.permute(0, -1, *range(1, target.dim())).float()
        topo_loss = self.topo(net_output, target_onehot)
        
        return self.weight_dice * dc_loss + self.weight_topo * topo_loss

class DC_SkelREC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, soft_skelrec_kwargs, weight_dice=1, weight_srec=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):

        super(DC_SkelREC_loss, self).__init__()

        self.weight_dice = weight_dice
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = SoftSkeletonRecallLoss(apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_skel = torch.where(mask, skel, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_skel = skel
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        srec_loss = self.srec(net_output, target_skel, loss_mask=mask) \
            if self.weight_srec != 0 else 0

        result = self.weight_dice * dc_loss + self.weight_srec * srec_loss
        return result


class DC_and_CE_SCloss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, sc_kwargs, weight_ce=1, weight_dice=1, weight_sc=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_SCloss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_sc = weight_sc
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.sc = MultiClassOneVsRestSCLoss(**sc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        sc_loss = self.sc(net_output,target)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_sc *sc_loss
        return result

class DC_and_CE_Clloss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cl_kwargs, weight_ce=1, weight_dice=1, weight_cl=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_Clloss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cl = weight_cl
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cl = ClDiceLoss(**cl_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        cl_loss = self.cl(net_output,target)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cl *cl_loss
        return result

class DC_and_BettiMatchingLoss_CE(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_topo=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_topo = weight_topo
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.topo = BettiMatchingLoss(softmax=True, use_base_loss=False, num_processes=os.cpu_count())
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # Dice Loss
        dc_loss = self.dc(net_output, target)

        # CE Loss
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            ce_loss = self.ce(net_output, target[:, 0])
        else:
            ce_loss = self.ce(net_output, target[:, 0])

        # Topo Loss
        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target_onehot = F.one_hot(target.long(), num_classes=net_output.shape[1])
        target_onehot = target_onehot.permute(0, -1, *range(1, target.dim())).float()
        topo_loss = self.topo(net_output, target_onehot)

        # Combine Losses
        return (
            self.weight_ce * ce_loss +
            self.weight_dice * dc_loss +
            self.weight_topo * topo_loss
        )

class DC_and_WassersteinLoss_CE(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_topo=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_topo = weight_topo
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.topo = HutopoLoss(softmax=True, use_base_loss=False, num_processes=os.cpu_count())
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # Dice Loss
        dc_loss = self.dc(net_output, target)

        # CE Loss
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            ce_loss = self.ce(net_output, target[:, 0])
        else:
            ce_loss = self.ce(net_output, target[:, 0])

        # Topo Loss
        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target_onehot = F.one_hot(target.long(), num_classes=net_output.shape[1])
        target_onehot = target_onehot.permute(0, -1, *range(1, target.dim())).float()
        topo_loss = self.topo(net_output, target_onehot)

        # Combine Losses
        return (
            self.weight_ce * ce_loss +
            self.weight_dice * dc_loss +
            self.weight_topo * topo_loss
        )

class DC_SkelREC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, soft_skelrec_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_srec=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_SkelREC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = SoftSkeletonRecallLoss(apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_skel = torch.where(mask, skel, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_skel = skel
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        srec_loss = self.srec(net_output, target_skel, loss_mask=mask) \
            if self.weight_srec != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss
        return result


class DC_and_WassersteinLoss_CE(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_topo=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_topo = weight_topo
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.topo = HutopoLoss(softmax=True, use_base_loss=False, num_processes=os.cpu_count())
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # Dice Loss
        dc_loss = self.dc(net_output, target)

        # CE Loss
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            ce_loss = self.ce(net_output, target[:, 0])
        else:
            ce_loss = self.ce(net_output, target[:, 0])

        # Topo Loss
        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target_onehot = F.one_hot(target.long(), num_classes=net_output.shape[1])
        target_onehot = target_onehot.permute(0, -1, *range(1, target.dim())).float()
        topo_loss = self.topo(net_output, target_onehot)

        # Combine Losses
        return (
            self.weight_ce * ce_loss +
            self.weight_dice * dc_loss +
            self.weight_topo * topo_loss
        )

class DC_SkelREC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, soft_skelrec_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_srec=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_SkelREC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = SoftSkeletonRecallLoss(apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            target_skel = torch.where(mask, skel, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_skel = skel
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        srec_loss = self.srec(net_output, target_skel, loss_mask=mask) \
            if self.weight_srec != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss
        return result
    

class DC_SoftGradientDiffTVLoss_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, weight_dice=1, weight_length=1, weight_tv=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_SoftGradientDiffTVLoss_loss, self).__init__()

        self.weight_dice = weight_dice
        self.weight_length = weight_length
        self.weight_tv = weight_tv
        self.ignore_label = ignore_label

        self.stv = SoftGradientDiffTVLoss(self.weight_length,self.weight_tv)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
    

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        stv_loss =  self.stv(net_output,target)

        result = self.weight_dice * dc_loss + stv_loss
        return result

class DC_CE_SoftGradientDiffTVLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, 
                 weight_dice=1, weight_ce=1, weight_length=1, weight_tv=1, 
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        """
        Combines Dice, CrossEntropy, and SoftGradientDiffTVLoss
        :param soft_dice_kwargs: args for Dice loss
        :param ce_kwargs: args for CrossEntropy loss
        :param weight_dice: Dice loss weight
        :param weight_ce: CrossEntropy loss weight
        :param weight_length: Soft Gradient Magnitude loss weight
        :param weight_tv: Difference Mask TV loss weight
        :param ignore_label: label index to ignore
        :param dice_class: Dice loss class
        """
        super(DC_CE_SoftGradientDiffTVLoss, self).__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_length = weight_length
        self.weight_tv = weight_tv
        self.ignore_label = ignore_label

        # Loss functions
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.stv = SoftGradientDiffTVLoss(self.weight_length, self.weight_tv)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        :param net_output: raw logits (batch, C, H, W)
        :param target: integer GT mask (batch, H, W), values {0,1,...}
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Dice Loss
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        # CrossEntropy Loss
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # SoftGradientDiffTVLoss (includes length + diff-TV)
        stv_loss = self.stv(net_output, target)

        # Combine losses
        total_loss = self.weight_dice * dc_loss + self.weight_ce * ce_loss + stv_loss
        return total_loss


class DC_and_CE_ClIoULoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, clioU_kwargs,
                 weight_ce=1, weight_dice=1, weight_cl=1, ignore_label=None,
                 dice_class=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_cl = weight_cl
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cliou = ClIoULoss(**clioU_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.dim() == 4 and target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice else 0
        ce_loss = self.ce(net_output, target[:, 0]) if self.weight_ce and (self.ignore_label is None or num_fg > 0) else 0
        cl_loss = self.cliou(net_output, target) if self.weight_cl else 0

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cl * cl_loss
    


class DC_and_ClIoULoss(nn.Module):
    def __init__(self, soft_dice_kwargs, clioU_kwargs,
                 weight_dice=1, weight_cl=1, ignore_label=None,
                 dice_class=None):
        super().__init__()

        self.weight_dice = weight_dice
        self.weight_cl = weight_cl
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cliou = ClIoULoss(**clioU_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.dim() == 4 and target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice else 0
        
        cl_loss = self.cliou(net_output, target) if self.weight_cl else 0

        return self.weight_dice * dc_loss + self.weight_cl * cl_loss
    

class TopologyAwareLoss(nn.Module):
    def __init__(self, soft_dice_kwargs,soft_skelrec_kwargs, topology_kwargs, weight_dice=1, weight_recall=0.5,
                dice_class=MemoryEfficientSoftDiceLoss):
        """
        Dice Loss에 Skeleton Recall과 Spurious Branch Penalty를 결합한 Loss.
        Penalty의 가중치(lambda)는 epoch에 따라 내부적으로 스케줄링됩니다.

        Args:
            soft_dice_kwargs (dict): SoftDiceLoss에 전달될 인자.
            topology_kwargs (dict): Recall 및 Penalty Loss와 스케줄러에 사용될 인자.
                                    (예: {'iterations': 10, 'total_epochs': 300, ...})
            weight_dice (float): Dice Loss의 고정 가중치.
            weight_recall (float): Skeleton Recall Loss의 고정 가중치.
            dice_class: 사용할 Dice Loss 클래스.
        """
        super(TopologyAwareLoss,self).__init__()
        self.weight_dice = weight_dice
        self.weight_recall = weight_recall

        # 스케줄러 파라미터 추출
        self.total_epochs = topology_kwargs.get('total_epochs', 300)
        self.start_lambda = topology_kwargs.get('start_lambda', 0.1)
        self.end_lambda = topology_kwargs.get('end_lambda', 1.0)
        self.ramp_up_fraction = topology_kwargs.get('ramp_up_fraction', 0.4)

        # Loss 함수 인스턴스 생성
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = SoftSkeletonRecallLoss(apply_nonlin=softmax_helper_dim1,**soft_skelrec_kwargs)
        self.spen = SpuriousBranchPenaltyLoss(iterations=topology_kwargs.get('iterations', 10),apply_nonlin=softmax_helper_dim1)

    def get_lambda_scheduler(self, current_epoch):
        ramp_up_epochs = int(self.total_epochs * self.ramp_up_fraction)
        if current_epoch < ramp_up_epochs:
            progress = float(current_epoch) / float(ramp_up_epochs)
            return self.start_lambda + (self.end_lambda - self.start_lambda) * progress
        else:
            return self.end_lambda

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor, current_epoch: int):
        # Dice Loss 계산
        dc_loss = self.dc(net_output, target) if self.weight_dice > 0 else 0

        # Topology-aware Loss 계산
        srec_loss = self.srec(net_output, skel) if self.weight_recall > 0 else 0
        
        # Penalty Loss는 항상 계산 (lambda가 0일 수 있으므로)
        spen_loss = self.spen(net_output, target)

        # 현재 epoch에 맞는 lambda 값 가져오기
        lambda_penalty = self.get_lambda_scheduler(current_epoch)

        # 최종 Loss 결합
        result = self.weight_dice * dc_loss + \
                self.weight_recall * srec_loss + \
                lambda_penalty * spen_loss
        
        return result
        

class DC_and_CE_TopologyAwareLoss(TopologyAwareLoss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, soft_skelrec_kwargs, topology_kwargs, weight_ce=1, weight_dice=1, 
                 weight_recall=0.5, dice_class=MemoryEfficientSoftDiceLoss):
        """
        TopologyAwareLoss에 Cross-Entropy Loss를 추가한 버전.
        """
        # 부모 클래스(TopologyAwareLoss)의 초기화 메소드 호출
        super().__init__(soft_dice_kwargs,soft_skelrec_kwargs, topology_kwargs, weight_dice, weight_recall, dice_class)
        
        self.weight_ce = weight_ce
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor, current_epoch: int):
        # 부모 클래스의 forward를 호출하여 Dice, Recall, Penalty Loss 계산
        topology_loss = super().forward(net_output, target, skel, current_epoch)
        
        # CE Loss 계산
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce > 0 else 0
        
        # CE Loss와 부모 클래스에서 계산된 Loss를 결합
        result = self.weight_ce * ce_loss + topology_loss
        return result