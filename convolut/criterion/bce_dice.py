from typing import Optional

import torch
from torch import nn

from ..settings import (
    CRITERION_BCEDICE_EPS,
    CRITERION_BCEDICE_THRESHOLD,
    CRITERION_BCEDICE_BCE_WEIGHT,
    CRITERION_BCEDICE_DICE_WEIGHT
)
from .dice import DiceLoss


class BCEDiceLoss(nn.Module):
    def __init__(self,
                 eps: float = CRITERION_BCEDICE_EPS,
                 threshold: float = CRITERION_BCEDICE_THRESHOLD,
                 bce_weight: float = CRITERION_BCEDICE_BCE_WEIGHT,
                 dice_weight: float = CRITERION_BCEDICE_DICE_WEIGHT,
                 activation: Optional[nn.Module] = nn.Sigmoid,
                 ):
        super().__init__()

        if bce_weight == 0 and dice_weight == 0:
            raise ValueError(
                "Both bce_wight and dice_weight cannot be "
                "equal to 0 at the same time."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.bce_weight != 0:
            self.bce_loss = nn.BCEWithLogitsLoss()

        if self.dice_weight != 0:
            self.dice_loss = DiceLoss(eps=eps, threshold=threshold, activation=activation)

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.bce_weight == 0:
            return self.dice_weight * self.dice_loss(output, target=target)

        if self.dice_weight == 0:
            return self.bce_weight * self.bce_loss(output, target)

        dice = self.dice_weight * self.dice_loss(output, target)
        bce = self.bce_weight * self.bce_loss(output, target)

        return dice + bce
