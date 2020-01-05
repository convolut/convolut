from typing import Optional

import torch
from torch import nn
from ..utils.dice import dice
from ..settings import (
    CRITERION_DICE_EPS,
    CRITERION_DICE_THRESHOLD
)


class DiceLoss(nn.Module):
    def __init__(self,
                 eps: float = CRITERION_DICE_EPS,
                 threshold: float = CRITERION_DICE_THRESHOLD,
                 activation: Optional[nn.Module] = nn.Sigmoid(),
                 ):
        super().__init__()

        self.eps = eps
        self.threshold = threshold
        self.activation = activation

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        value = dice(output, target, self.eps, self.threshold, self.activation)

        return 1 - value
