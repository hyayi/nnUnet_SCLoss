import numpy as np
import torch
import torch.nn as nn
from typing import Callable
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss

from nnunetv2.training.loss.soft_skeleton import SoftSkeletonize
import cupy as cp
from cucim.core.operations import morphology


class CLossTopo(nn.Module):
    def __init__(self, apply_nonlin: Callable = softmax_helper_dim1, batch_dice: bool = False, do_bg: bool = False, smooth: float = 1e-5,
                 ddp: bool = False,iter_=50, beta=1):
        """
        Calculates the Closs term that addresses the topological critical pixels of the predicted image and ground truth image
        Args:
            pred:   The likelihood pytorch tensor for neural networks.
            gt:   The groundtruth of pytorch tensor.
        Returns:
            ClossTopo:   The Closs value belonging to the topological term (tensor)
        """
        super(CLossTopo, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)
        self.beta = beta #for F1 score weighting, recall weighted beta times more important, optional use eg. for CLoss(dice)

    def forward(self, x, y, loss_mask=None):
        """
        Input shape: (batchsize,number_of_classes,z_dim,y_dim,x_dim)
        x: pred      number_of_classes is 2 in binary case, includes background
        y: gt       number_of_classes is 1 in binary case
        """

        x_logits=x  #presumed example shape 2,2,200,1024,1024
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x) #preserves shape 2,2,200,1024,1024, normalises to positive Intervall [0,1]
        pred_binary = torch.argmax(x, dim=1) #pred shape = 2,200,1024,1024 contains pred idx analog gt
        pred_binary = pred_binary.unsqueeze(1) # 2,1,200,1024,1024, binary

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:] # shape 2,1,200,1024,1024

        #skelet gt-pred
        with torch.no_grad():
            skel_gt = self.soft_skeletonize(y) #float of binary tensor, otherwise error for boolean values in soft_skeletonize
        diff_skel_gt_minus_pred = skel_gt - pred_binary
        diff_skel_gt_minus_pred[diff_skel_gt_minus_pred < 0]=0 #2,1,200,1024,1024,

        #skelet gt-pred
        with torch.no_grad():
            skel_pred = self.soft_skeletonize(x)
        diff_skel_pred_minus_gt = skel_pred - y
        diff_skel_pred_minus_gt[diff_skel_pred_minus_gt < 0]=0 #2,1,200,1024,1024,

        #well predicted for gt&pred skelet
        well_pred_gt_skel = skel_gt - diff_skel_gt_minus_pred 
        well_pred_gt_skel[well_pred_gt_skel<0]=0
        well_preds_pred_skel = skel_pred - diff_skel_pred_minus_gt 
        well_preds_pred_skel[well_pred_gt_skel<0]=0

        #find context pxl around crit pxls in gt&pred skel
        cp_diff_skel_gt_minus_pred_gt_skel = cp.asarray(diff_skel_gt_minus_pred.cpu().detach().numpy())
        cp_well_pred_skel_gt_skel = cp.asarray(well_pred_gt_skel.cpu().detach().numpy())

        cp_diff_skel_pred_minus_gt_pred_skel = cp.asarray(diff_skel_pred_minus_gt.cpu().detach().numpy())
        cp_well_pred_skel_pred_skel = cp.asarray(well_preds_pred_skel.cpu().detach().numpy())

        #prepare empty storage
        cp_dist_crit_pxls_gt_skel = cp.empty_like(cp_diff_skel_gt_minus_pred_gt_skel)
        cp_dist_already_good_pxls_gt_skel = cp.empty_like(cp_well_pred_skel_gt_skel)

        cp_dist_crit_pxls_pred_skel = cp.empty_like(cp_diff_skel_pred_minus_gt_pred_skel)
        cp_dist_already_good_pxls_pred_skel = cp.empty_like(cp_well_pred_skel_pred_skel)


        for i in range(x.shape[0]):
            #distance transforms gt skel
            cp_dist_crit_pxls_gt_skel[i,0] = morphology.distance_transform_edt(1-cp_diff_skel_gt_minus_pred_gt_skel[i,0]) #morph return shape: 200,1024,1024
            cp_dist_already_good_pxls_gt_skel[i,0] = morphology.distance_transform_edt(1-cp_well_pred_skel_gt_skel[i,0]) #morph return shape: 200,1024,1024

            #distance transforms pred skel
            cp_dist_crit_pxls_pred_skel[i,0] = morphology.distance_transform_edt(1-cp_diff_skel_pred_minus_gt_pred_skel[i,0]) #morph return shape: 200,1024,1024
            cp_dist_already_good_pxls_pred_skel[i,0] = morphology.distance_transform_edt(1-cp_well_pred_skel_pred_skel[i,0]) #morph return shape: 200,1024,1024


        #compare distances to find context pxl around gt&pred skeletons
        all_pxl_nearest_critical_pxls_gt_skel = cp.where(cp_dist_already_good_pxls_gt_skel > cp_dist_crit_pxls_gt_skel, 1,0)
        all_pxl_nearest_critical_pxls_pred_skel = cp.where(cp_dist_already_good_pxls_pred_skel > cp_dist_crit_pxls_pred_skel, 1,0)
        
        cp_y= cp.asarray(y.cpu().detach().numpy())
        cp_gt_focused_around_crit_pxls_gt_skel = all_pxl_nearest_critical_pxls_gt_skel * cp_y

        #where foreground class dominant
        cp_pred_binary= cp.asarray(pred_binary.cpu().detach().numpy())
        cp_pred_focused_around_crit_pxls_pred_skel = all_pxl_nearest_critical_pxls_pred_skel * cp_pred_binary

        #convert to torch
        torch_gt_focused_around_crit_pxls_gt_skel = torch.from_numpy(cp_gt_focused_around_crit_pxls_gt_skel.get()).cuda()
        torch_pred_focused_around_crit_pxls_pred_skel = torch.from_numpy(cp_pred_focused_around_crit_pxls_pred_skel.get()).cuda()

        #Dice loss (soft dice calculation, similar to cldice but returns -dc analogue to MemoryEfficientSoftDiceLoss of nnunet)
        tprec = (torch.sum(torch.multiply(torch_pred_focused_around_crit_pxls_pred_skel, y))+self.smooth)/(torch.sum(torch_pred_focused_around_crit_pxls_pred_skel)+self.smooth)    
        tsens = (torch.sum(torch.multiply(torch_gt_focused_around_crit_pxls_gt_skel, x))+self.smooth)/(torch.sum(torch_gt_focused_around_crit_pxls_gt_skel)+self.smooth)
        beta_2 = self.beta**2
        dice_result = -(1+beta_2)*(tprec*tsens)/((beta_2*tprec)+tsens)

        #CE loss
        ce_loss = RobustCrossEntropyLoss()
        all_crit_pxls = torch.logical_or(torch_pred_focused_around_crit_pxls_pred_skel, torch_gt_focused_around_crit_pxls_gt_skel) #2,1,200,1024,1024; could restrain to just gt pxls
        x_part = x_logits * all_crit_pxls #shape x_logits is (2,2,200,1024,1024)*all_crit_pxls...(2,1,200,1024,1024)=(2,2,200,1024,1024) because shape broadcasting
        y_part = y * all_crit_pxls
        ce_result = ce_loss(x_part, y_part)

        return dice_result + ce_result


class CustomLoss_With_DiceCe(nn.Module):
    def __init__(self, weight_ce=1, weight_dice=1, weight_custom=1, ignore_label=None, ce_class=RobustCrossEntropyLoss,
                 dice_class=MemoryEfficientSoftDiceLoss, custom_loss_class=CLossTopo, gamma=0.5):
        """
        Combines Custom pixelwise loss with standard Dice&Ce loss. Weights for CE, Dice and custom loss do not need to sum to one. You can set whatever you want.
        """
        super(CustomLoss_With_DiceCe, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.weight_custom = weight_custom
        self.gamma = gamma
        self.ce = ce_class()
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=False, smooth=1e-5, ddp=False)
        self.custom_loss = custom_loss_class()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c,( z,) y, x with c=1
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        custom_loss = self.custom_loss(net_output, target, loss_mask=mask) \
            if self.weight_custom != 0 else 0

        standard_part = (1.0 - self.gamma) * (self.weight_ce * ce_loss + self.weight_dice * dc_loss)
        topo_part = self.gamma * (self.weight_custom * custom_loss)

        return standard_part + topo_part

class CLoss(CustomLoss_With_DiceCe):
    def __init__(self, weight_ce=1, weight_dice=1, weight_custom=1, ce_class = RobustCrossEntropyLoss, dice_class=MemoryEfficientSoftDiceLoss, custom_loss_class=CLossTopo, gamma=0.5):
        """
            Creates the total Closs loss function for loss calculation of the predicted image and ground truth image.
            Args:
                weight_ce:   Weight of cross entropy loss. Doesn't need to sum to one.
                weight_dice:   Weight of dice loss. Doesn't need to sum to one.
                weight_custom:   Weight of custom loss. Doesn't need to sum to one.
                ce_class:   Cross entropy loss.
                dice_class:   Dice loss.
                custom_loss_class:   Custom loss.
                gamma:   Weight to adjust impact of the topology-sensitive-term, analogue to our paper.
            Returns:
                Closs:   The Closs function
        """
        super().__init__(weight_ce=weight_ce, weight_dice=weight_dice, weight_custom=weight_custom, ce_class = ce_class, dice_class=dice_class, custom_loss_class=custom_loss_class, gamma=gamma)
