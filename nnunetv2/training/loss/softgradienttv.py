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

    def soft_gradient_magnitude(self,mask: torch.Tensor):
        """
        Compute soft gradient magnitude using Sobel filters
        mask: (B, C, H, W) tensor
        """
        B, C, H, W = mask.shape

        # Sobel filters
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]]], device=mask.device, dtype=mask.dtype).repeat(C, 1, 1, 1)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]]]], device=mask.device, dtype=mask.dtype).repeat(C, 1, 1, 1)

        # Depthwise convolution
        grad_x = F.conv2d(mask, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(mask, sobel_y, padding=1, groups=C)

        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-5)
        return grad_mag

    def total_variation(self, mask):
        """
        Compute total variation loss
        """
        tv_h = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
        tv_w = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])
        tv = (torch.sum(tv_h) + torch.sum(tv_w))/ mask.shape[0]
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
        target_tube = (target == 1).float()  # (batch, 1, H, W)


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

