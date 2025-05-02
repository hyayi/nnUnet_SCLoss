import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class BinarySCLoss(nn.Module):
    def __init__(self, k: int = 2, alpha: float = 1.0) -> None:
        """
        Spatial Coherence Loss (Binary)

        Args:
            k (int): 주변 픽셀 거리 단계 (1~k)
            alpha (float): mutual / pairwise 비중
        """
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.eps = 1e-6

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: [B, 1, H, W] - 로짓 출력
            target: [B, 1, H, W] - 바이너리 정답 (0 또는 1)

        Returns:
            loss (Tensor): 최종 스칼라 손실값
        """
        B, _ , H, W = pred.size()
        total_loss = torch.zeros(B, H * W, device=pred.device)

        # BCE with logits → 안정성 확보
        # 확률값은 sigmoid(pred)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # [B, 1, H, W]

        for i in range(1, self.k + 1):
            kernel_size = 2 * i + 1
            patch_area = kernel_size ** 2
            center_idx = patch_area // 2

            # unfold 예측/정답
            unfold_pred = F.unfold(pred, kernel_size=kernel_size, padding=i).view(B, patch_area, -1)   # [B, P, H*W]
            unfold_target = F.unfold(target, kernel_size=kernel_size, padding=i).view(B, patch_area, -1)

            # mask: 경계 padding 구역 제외
            valid_mask = F.unfold(torch.ones_like(target), kernel_size=kernel_size, padding=i)
            valid_mask = (valid_mask == 0).view(B, patch_area, -1)  # [B, P, H*W]

            # 중심 픽셀
            center_pred = unfold_pred[:, center_idx:center_idx+1, :]   # [B, 1, H*W]
            center_target = unfold_target[:, center_idx:center_idx+1, :]


            # mutual response 계산
            joint_prob = (center_pred * unfold_pred)
            label_product = center_target * unfold_target
            mutual = F.binary_cross_entropy_with_logits(joint_prob,label_product,reduction='none')
            # if torch.isnan(mutual).any():
            #     print(f"label_product {torch.isnan(label_product).any()} joint_prob {torch.isnan(joint_prob.sigmoid()).any()} center_pred {torch.isnan(center_pred).any()},unfold_pred {torch.isnan(unfold_pred).any()}")
            # pairwise 계산
            pairwise = torch.exp(-center_pred.sigmoid() * unfold_pred.sigmoid())  # [B, P, H*W]
            # dot = (center_pred * unfold_pred)
            # pairwise = F.softmax(dot, dim=1)  # [B, P, H*W]

            # BCE도 unfold처럼 확장
            bce_unfold = bce.repeat(1, patch_area, 1, 1).view(B, patch_area, -1)

            # 총합 손실 계산
            loss = bce_unfold / (mutual + self.alpha * pairwise + self.eps)
            loss = loss.masked_fill(valid_mask, 1.0)

            total_loss += ((1 / 2) ** (i - 1)) * (loss.sum(dim=1) / patch_area)  # [B, H*W]

        return total_loss.mean()

class MultiClassOneVsRestSCLoss(nn.Module):
    """
    멀티 클래스 분류에 대해 클래스별로 BinarySCLoss를 적용하여 평균
    """
    def __init__(self, k: int = 2, alpha: float = 1.0,target_classes:list =[1,2]):
        super().__init__()
        self.sc_loss = BinarySCLoss(k=k, alpha=alpha)
        self.target_classes = target_classes

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred: [B, C, H, W] - 로짓 (softmax 적용 전)
        target: [B, H, W] - 정수형 클래스 인덱스
        """
        if target.ndim == pred.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]

        B, C, H, W = pred.shape
        total_loss = 0.0

        for c in self.target_classes:
            # 현재 클래스에 대해 One-vs-Rest binary mask 생성
            pred_c = pred[:, c:c+1, :, :]                        # [B, 1, H, W]
            target_c = (target == c).float().unsqueeze(1)        # [B, 1, H, W]
            
            loss_c = self.sc_loss(pred_c, target_c)
            total_loss += loss_c

        return total_loss / len(self.target_classes)


# class MultiClassSCLoss(nn.Module):
#     def __init__(self, k: int = 2, alpha: float = 1.0) -> None:
#         """
#         Multi-class Spatial Coherence Loss (dot product + binary BCE 기반)

#         Args:
#             k (int): patch 거리 범위
#             alpha (float): mutual vs pairwise 비중 조절
#         """
#         super().__init__()
#         self.k = k
#         self.alpha = alpha
#         self.eps = 1e-8

#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             pred: [B, C, H, W] - raw logits
#             target: [B, H, W] - 정수형 클래스 인덱스

#         Returns:
#             loss: scalar tensor
#         """
#         B, C, H, W = pred.shape  # [B, C, H, W]

#         if target.ndim == pred.ndim:
#             assert target.shape[1] == 1
#             target = target[:, 0]

#         if target.dtype != torch.long:
#             target = target.long()

#         ce = F.cross_entropy(pred, target, reduction='none')  # [B, H, W]
#         ce = ce.view(B, 1, H, W)  # [B, 1, H, W]

#         total_loss = torch.zeros(B, H * W, device=pred.device)  # [B, H*W]

#         for i in range(1, self.k + 1):
#             kernel_size = 2 * i + 1
#             patch_area = kernel_size ** 2
#             center_idx = patch_area // 2

#             unfold_logits = F.unfold(pred, kernel_size=kernel_size, padding=i).view(B, C, patch_area, -1)  # [B, C, P, H*W]
#             center_logits = unfold_logits[:, :, center_idx:center_idx+1, :]  # [B, C, 1, H*W]

#             # dot similarity (cosine-like)
#             sim_logits = center_logits * unfold_logits   # [B, P, H*W]

#             # 라벨 unfold
#             target_unfold = F.unfold(target.unsqueeze(1).float(), kernel_size=kernel_size, padding=i).view(B, patch_area, -1).long()  # [B, P, H*W]
#             center_label = target_unfold[:, center_idx, :]  # [B, H*W]

#             # 클래스별 BCE 계산 (background 제외)
#             mutual_list = []
#             for cls in range(1, C):  # background 제외
#                 cls_mask = (target_unfold == cls)  # [B, P, H*W]
#                 center_cls_mask = (center_label == cls).unsqueeze(1)  # [B, 1, H*W]
#                 combined_mask = cls_mask & center_cls_mask  # [B, P, H*W]
#                 binary_labels = combined_mask.float()

#                 cls_sim_logits = sim_logits[:, cls, :, :]  # [B, P, H*W] - 클래스 cls에 해당하는 유사도만 추출
#                 cls_bce = F.binary_cross_entropy_with_logits(cls_sim_logits, binary_labels, reduction='none') # [B, P, H*W]
#                 mutual_list.append(cls_bce.unsqueeze(1))  # [B, 1, P, H*W]

#             mutual = torch.cat(mutual_list, dim=1)  # [B, C-1, P, H*W]
#             mutual = mutual.mean(dim=1, keepdim=True)  # [B, 1, P, H*W]

#             # pairwise similarity (softmax normalized)
#             pairwise = torch.exp(-sim_logits.sum(dim=1)).unsqueeze(1)  # [B, 1, P, H*W]
#             # unfold CE
#             ce_unfold = F.unfold(ce, kernel_size=kernel_size, padding=i).view(B, 1, patch_area, -1)  # [B, 1, P, H*W]

#             # valid mask
#             valid_mask = F.unfold(torch.ones((B, 1, H, W), device=pred.device), kernel_size=kernel_size, padding=i)
#             valid_mask = (valid_mask == 0).view(B, 1, patch_area, -1)  # [B, 1, P, H*W]

#             # final loss
#             denom = mutual + self.alpha * pairwise + self.eps
#             loss = ce_unfold / denom
#             loss = loss.masked_fill(valid_mask, 1.0)

#             if torch.isnan(loss).any():
#                 print(f"mutual NaN: {torch.isnan(mutual).any()}, pairwise NaN: {torch.isnan(pairwise).any()}")

#             total_loss += ((1 / 2) ** (i - 1)) * (loss.sum(dim=2).squeeze(1) / patch_area)  # [B, H*W]

#         return total_loss.mean()  # scalar

    
# class MultiClassSCLoss(nn.Module):
#     def __init__(self, k: int = 2, alpha: float = 1.0) -> None:
#         """
#         Multi-class Spatial Coherence Loss (dot product + binary BCE 기반)

#         Args:
#             k (int): patch 거리 범위
#             alpha (float): mutual vs pairwise 비중 조절
#         """
#         super().__init__()
#         self.k = k
#         self.alpha = alpha
#         self.eps = 1e-8

#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             pred: [B, C, H, W] - raw logits
#             target: [B, H, W] - 정수형 클래스 인덱스

#         Returns:
#             loss: scalar tensor
#         """
#         B, C, H, W = pred.shape  # [B, C, H, W]

#         if target.ndim == pred.ndim:
#             assert target.shape[1] == 1
#             target = target[:, 0]

#         if target.dtype != torch.long:
#             target = target.long()

#         # CrossEntropyLoss 계산
#         ce = F.cross_entropy(pred, target, reduction='none')  # [B, H, W]
#         ce = ce.view(B, 1, H, W)  # [B, 1, H, W]

#         total_loss = torch.zeros(B, H * W, device=pred.device)  # [B, H*W]

#         for i in range(1, self.k + 1):
#             kernel_size = 2 * i + 1
#             patch_area = kernel_size ** 2
#             center_idx = patch_area // 2

#             # unfold logits: [B, C, P, H*W]
#             unfold_logits = F.unfold(pred, kernel_size=kernel_size, padding=i).view(B, C, patch_area, -1) 
#             center_logits = unfold_logits[:, :, center_idx:center_idx+1, :]  # [B, H*W, 1, C ] x[B, H*W , C ,P ]

#             # dot similarity using matmul: [B, 1, P, H*W]
#             #[B, C,1,H*W] = > [B, H*W,1,C] 
#             #[B,C ,P, H*W ] -> [B,H*W,P,C] -> [B, H*W,C,P]
#             #[B, H*W,1,P] -> [B,P,1,H*W] -> [B,1,P,H*W]
#             sim_logits = torch.matmul(center_logits.transpose(1,3), unfold_logits.transpose(1,3).transpose(2,3))  # [B, 1, P, H*W]
#             sim_logits = sim_logits.transpose(1,3).transpose(1,2)  # [B, P, H*W]
#             sim_logits = sim_logits.squeeze(1)
#             print(sim_logits.mean())

#             # sigmoid로 유사도 해석

#             # 라벨 unfold: [B, P, H*W]
#             target_unfold = F.unfold(target.unsqueeze(1).float(), kernel_size=kernel_size, padding=i).view(B, patch_area, -1).long()  # [B, P, H*W]
#             center_label = target_unfold[:, center_idx, :]  # [B, H*W]
#             label_mask = (target_unfold == center_label.unsqueeze(1)).float()  # [B, P, H*W]

#             # binary BCE loss
#             mutual = F.binary_cross_entropy_with_logits(sim_logits, label_mask, reduction='none')  # [B, P, H*W]
#             mutual = mutual.unsqueeze(1)  # [B, 1, P, H*W]

#             # pairwise: softmax로 정규화된 유사도 weight
#             pairwise = torch.exp(-sim_logits).unsqueeze(1)  # [B, 1, P, H*W]

#             # CE unfold: [B, 1, P, H*W]
#             ce_unfold = F.unfold(ce, kernel_size=kernel_size, padding=i).view(B, 1, patch_area, -1)

#             # valid mask
#             valid_mask = F.unfold(torch.ones((B, 1, H, W), device=pred.device), kernel_size=kernel_size, padding=i)
#             valid_mask = (valid_mask == 0).view(B, 1, patch_area, -1)  # [B, 1, P, H*W]

#             # final loss
#             denom = mutual + self.alpha * pairwise + self.eps  # [B, 1, P, H*W]
#             loss = ce_unfold / denom
#             loss = loss.masked_fill(valid_mask, 1.0)
#             if torch.isnan(loss).any():
#                 print(f"mutual {torch.isnan(mutual).any()} , pairwise{torch.isnan(pairwise).any()} ")

#             total_loss += ((1 / 2) ** (i - 1)) * (loss.sum(dim=2).squeeze(1) / patch_area)  # [B, H*W]

#         return total_loss.mean()  # scalar

#     # def soft_cross_entropy(self,logits, target, reduction='mean', eps=1e-8):
#     #     log_probs = F.log_softmax(logits, dim=1)  # 수치 안정성 확보
#     #     # log_probs = torch.clamp(log_probs, min=torch.log(torch.tensor(eps)))  # 선택적 클램핑
#     #     loss = -torch.sum(target * log_probs, dim=1)  # 클래스 축 합산 → [B, H, W]
#     #     if torch.isnan(loss).any():
#     #         print(torch.isnan(log_probs))

        
#     #     if reduction == 'mean':
#     #         return loss.mean()
#     #     elif reduction == 'sum':
#     #         return loss.sum()
#     #     else:
#     #         return loss  # [B, H, W]

