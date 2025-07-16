import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftGradientDiffTVLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        """
        alpha: Soft Gradient Magnitude Loss weight
        beta: Difference Mask Total Variation Loss weight
        """
        super(SoftGradientDiffTVLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def soft_gradient_magnitude(self, mask):
        """
        Compute soft gradient magnitude
        """
        grad_x = mask[:, :, :, 1:] - mask[:, :, :, :-1]  # (B, C, H, W-1)
        grad_y = mask[:, :, 1:, :] - mask[:, :, :-1, :]  # (B, C, H-1, W)

        # Pad to match size
        grad_x = F.pad(grad_x, (0, 1))  # pad width to match original W
        grad_y = F.pad(grad_y, (0, 0, 0, 1))  # pad height to match original H

        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-5)
        return grad_mag

    def total_variation(self, mask):
        """
        Compute total variation loss
        """
        tv_h = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
        tv_w = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
        tv = torch.sum(tv_h + tv_w) / mask.shape[0]
        return tv

    def forward(self, pred, target):
        """
        pred: raw logits (batch, 2, H, W)
        target: integer GT mask (batch, H, W), values: {0,1}
        """
        # Softmax over class dimension
        pred_softmax = F.softmax(pred, dim=1)  # (batch, 2, H, W)

        # Tube class only
        pred_tube = pred_softmax[:, 1:2, :, :]         # (batch, 1, H, W)
        target_tube = (target == 1).float().unsqueeze(1)  # (batch, 1, H, W)


        # Soft Gradient Magnitude Loss
        grad_mag_pred = self.soft_gradient_magnitude(pred_tube)
        grad_mag_gt = self.soft_gradient_magnitude(target_tube)

        pred_lengths = torch.sum(grad_mag_pred, dim=(1,2,3))  # (batch_size,)
        gt_lengths = torch.sum(grad_mag_gt, dim=(1,2,3))      # (batch_size,)

        length_loss = torch.abs(pred_lengths - gt_lengths) / (gt_lengths + 1e-5)
        length_loss = length_loss.mean()

        # Difference Mask Total Variation Loss
        diff_mask = torch.abs(pred_tube - target_tube)
        diff_tv_loss = self.total_variation(diff_mask)

        # Combine losses
        total_loss = self.alpha * length_loss + self.beta * diff_tv_loss
        return total_loss

