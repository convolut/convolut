from typing import Optional

import torch
from torch import nn
from ..utils.dice import dice


class DiceLoss(nn.Module):
    def __init__(self,
                 eps: float = 1e-7,
                 threshold: float = None,
                 activation: Optional[nn.Module] = nn.Sigmoid(),
                 ):
        super().__init__()

        self.eps = eps
        self.threshold = threshold
        self.activation = activation

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        value = dice(output, target, self.eps, self.threshold, self.activation)

        return 1 - value
