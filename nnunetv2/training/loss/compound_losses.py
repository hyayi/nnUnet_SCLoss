import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.scloss import BinarySCLoss,MultiClassOneVsRestSCLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.cldice import ClDiceLoss
from torch import nn
from torch import Tensor
from topolosses.losses.hutopo import HutopoLoss
from topolosses.losses.betti_matching import BettiMatchingLoss
import torch.nn.functional as F

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
        dice_class=SoftDiceLoss
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
    

# class DC_and_MSC_loss(nn.Module):
#     def __init__(
#         self,
#         soft_dice_kwargs: dict,
#         sc_kwargs: dict,
#         weight_dice: float = 1.0,
#         weight_sc: float = 1.0,
#         dice_class=SoftDiceLoss
#     ):
#         """
#         DC (Dice)와 SC (Spatial Coherence)를 합친 복합 손실함수

#         Args:
#             num_classes (int): 클래스 수
#             soft_dice_kwargs (dict): Dice 손실 설정 파라미터
#             sc_kwargs (dict): SC 손실 설정 파라미터 (k, alpha)
#             weight_dice (float): Dice 손실 가중치
#             weight_sc (float): SC 손실 가중치
#             do_bg (bool): SC에서 배경(클래스 0)을 제외할지 여부
#             dice_class: 사용할 Dice Loss 클래스
#         """
#         super().__init__()

#         self.weight_dice = weight_dice
#         self.weight_sc = weight_sc

#         self.dc = dice_class(apply_nonlin=softmax_helper_dim1,**soft_dice_kwargs)
#         self.sc = MultiClassSCLoss(**sc_kwargs)

#     def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
#         """
#         Args:
#             net_output (Tensor): [B, C, H, W] - 로짓 출력
#             target (Tensor): [B, H, W] - 정답 클래스 인덱스

#         Returns:
#             loss (Tensor): 최종 손실값
#         """
#         sc_loss = self.sc(net_output, target)
#         dice_loss = self.dc(net_output, target)

#         return self.weight_dice * dice_loss + self.weight_sc * sc_loss
    


class DC_and_CE_SCloss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, sc_kwargs, weight_ce=1, weight_dice=1, weight_sc=1, ignore_label=None,
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
    

class DC_and_SC_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs: dict,
        sc_kwargs: dict,
        weight_dice: float = 1.0,
        weight_sc: float = 1.0,
        dice_class=SoftDiceLoss
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

class DC_Clloss(nn.Module):
    def __init__(self, soft_dice_kwargs, cl_kwargs, weight_dice=1, weight_cl=1, ignore_label=None,
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


class CE_Clloss(nn.Module):
    def __init__(self,  ce_kwargs, cl_kwargs, weight_ce=1, weight_cl=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(CE_Clloss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_ce = weight_ce
        self.weight_cl = weight_cl
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
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

        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        cl_loss = self.cl(net_output,target)

        result = self.weight_ce * ce_loss + self.weight_cl *cl_loss
        return result

class DC_and_BettiMatchingLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, weight_topo=1, weight_dice=1,
                 dice_class=None):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_topo = weight_topo

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.topo = BettiMatchingLoss(softmax=True,use_base_loss=False,num_processes=4)

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
                 dice_class=None):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_topo = weight_topo

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.topo = HutopoLoss(softmax=True,use_base_loss=False,num_processes=2)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        dc_loss = self.dc(net_output, target)

        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target_onehot = F.one_hot(target.long(), num_classes=net_output.shape[1]) #(B,H,W,C)
        target_onehot = target_onehot.permute(0, -1, *range(1, target.dim())).float()
        topo_loss = self.topo(net_output, target_onehot)
        
        return self.weight_dice * dc_loss + self.weight_topo * topo_loss
