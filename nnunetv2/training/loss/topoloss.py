import torch
from BettiMatching import *
import torch.nn as nn

def compute_BettiMatchingLoss(t, softmax=True, relative=False, comparison='union', filtration='superlevel', construction='V'):
    if softmax:
        pred = torch.softmax(t[0],1)
        pred = pred[1]
    else:
        pred = t[0]
        pred = pred[1]
    if filtration != 'bothlevel':
        BM = BettiMatching(pred, t[1], relative=relative, comparison=comparison, filtration=filtration, construction=construction, training=True)
        loss = BM.loss()
    else:
        BM = BettiMatching(pred, t[1], relative=relative, comparison=comparison, filtration='superlevel', construction=construction, training=True)
        loss = BM.loss()
        BM = BettiMatching(pred, t[1], relative=relative, comparison=comparison, filtration='sublevel', construction=construction, training=True)
        loss += BM.loss()
    return loss


def compute_WassersteinLoss(t, softmax=True, relative=False, filtration='superlevel', construction='V', dimensions=[0,1]):
    if softmax:
        pred = torch.softmax(t[0],1)
        pred = pred[1]
    else:
        pred = t[0]
        pred = pred[1]
    WM = WassersteinMatching(pred, t[1], relative=relative, filtration=filtration, construction=construction, training=True)
    loss = WM.loss(dimensions=dimensions)
    return loss


def compute_ComposedWassersteinLoss(t, softmax=True, relative=False, filtration='superlevel', construction='V', comparison='union', dimensions=[0,1]):
    if softmax:
        pred = torch.softmax(t[0],1)
        pred = pred[1]
    else:
        pred = t[0]
        pred = pred[1]
    WM = ComposedWassersteinMatching(pred, t[1], relative=relative, filtration=filtration, construction=construction, comparison=comparison, training=True)
    loss = WM.loss(dimensions=dimensions)
    return loss


class BettiMatchingLoss(nn.Module):
    def __init__(
        self,
        relative=True,
        filtration='superlevel',
    ) -> None:
        super().__init__()
        self.relative = relative
        self.filtration = filtration

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for pair in zip(input,target):
            losses.append(compute_BettiMatchingLoss(pair, softmax=True, filtration=self.filtration, relative=self.relative))
        loss = torch.mean(torch.stack(losses))
        return loss

class WassersteinLoss(nn.Module):
    def __init__(
        self,
        relative=False,
        filtration='superlevel',
        dimensions=[0,1],
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.relative = relative
        self.filtration = filtration
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for pair in zip(input,target):
            losses.append(compute_WassersteinLoss(pair, softmax=True, filtration=self.filtration, relative=self.relative, dimensions=self.dimensions))
        loss = torch.mean(torch.stack(losses))
        return loss
