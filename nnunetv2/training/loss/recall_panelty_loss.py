import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import SoftSkeletonize # 필요에 따라 경로 수정

class SpuriousBranchPenaltyLoss(nn.Module):
    """
    잘못된 브랜치에 페널티를 부과하는 Loss 함수.
    GT가 one-hot이 아닌 레이블 맵 형태일 때를 지원합니다.
    """
    def __init__(self, iterations=10, smooth=1e-6, apply_nonlin=None, ddp: bool = False):
        super(SpuriousBranchPenaltyLoss, self).__init__()
        self.iterations = iterations
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin
        self.ddp = ddp
        self.soft_skeletonize = SoftSkeletonize(num_iter=iterations)

    def forward(self, net_output, gt_label_map):
        """
        Args:
            net_output (torch.Tensor): 모델의 원시 출력 (logits), shape: (B, C, H, W).
            gt_label_map (torch.Tensor): Ground Truth 레이블 맵, shape: (B, 1, H, W).
        """
        num_classes = net_output.shape[1]

        # --- GT 레이블 맵을 one-hot 형태로 변환 ---
        # (B, 1, H, W) -> (B, H, W)
        gt_labels = gt_label_map.squeeze(1).long()
        # (B, H, W) -> (B, H, W, C)
        gt_mask_one_hot = F.one_hot(gt_labels, num_classes=num_classes)
        # (B, H, W, C) -> (B, C, H, W)
        gt_mask = gt_mask_one_hot.permute(0, 3, 1, 2).float()
        
        # 1. Non-linearity 적용
        if self.apply_nonlin is not None:
            pred_proba = self.apply_nonlin(net_output)
        else:
            pred_proba = net_output
        
        # 2. 전경 채널 선택
        pred_proba_fg = pred_proba[:, 1:]
        gt_mask_fg = gt_mask[:, 1:]
        
        # ... (스켈레톤 생성 및 잘못된 브랜치 식별 로직은 동일) ...
        predicted_skeleton_fg = self.soft_skeletonize(pred_proba_fg)
        gt_background_mask_fg = 1 - gt_mask_fg
        spurious_branches = predicted_skeleton_fg * gt_background_mask_fg
        
        # 6. 페널티 계산
        numerator = torch.sum(spurious_branches, dim=[1, 2, 3])
        denominator = torch.sum(predicted_skeleton_fg, dim=[1, 2, 3])
        
        # ddp=True일 때만 AllGatherGrad 호출
        if self.ddp:
            from nnunetv2.utilities.ddp_allgather import AllGatherGrad
            numerator = AllGatherGrad.apply(numerator).sum(0)
            denominator = AllGatherGrad.apply(denominator).sum(0)
        
        penalty = (numerator + self.smooth) / (denominator + self.smooth)
        
        return penalty.mean()