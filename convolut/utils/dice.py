from typing import Optional

import torch
from torch import nn


def dice(
        output: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-7,
        threshold: float = None,
        activation: Optional[nn.Module] = nn.Sigmoid(),
):
    activation = activation if activation else lambda x: x

    output = activation(output)

    if threshold is not None:
        output = (output > threshold).float()

    intersection = torch.sum(target * output)
    union = torch.sum(target) + torch.sum(output)

    result = 2 * intersection / (union + eps)

    return result
