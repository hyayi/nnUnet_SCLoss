import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):

        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):

        return self.soft_skel(img)

class ClIoULoss(nn.Module):
    def __init__(self, num_iter=20, eps=1e-6):
        super().__init__()
        self.skel = SoftSkeletonize(num_iter=num_iter)
        self.eps = eps

    def forward(self, y_pred_logits, y_true):
        prob = F.softmax(y_pred_logits, dim=1)
        tube_pred = prob[:, 1:2, ...]
        y_true_oh = F.one_hot(y_true.long(), num_classes=2).permute(0,3,1,2).float()
        tube_gt = y_true_oh[:, 1:2, ...]

        skel_pred = self.skel(tube_pred)
        skel_gt = self.skel(tube_gt)

        inter = torch.min(skel_pred, skel_gt).sum(dim=(1,2,3))
        union = torch.max(skel_pred, skel_gt).sum(dim=(1,2,3)) + self.eps
        cliou = inter / union

        return (1 - cliou).mean()
