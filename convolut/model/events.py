from dataclasses import dataclass
from typing import Any

import torch
from decouple import Event
from torch import nn

from ..runner import Runner


@dataclass
class ModelInitEvent(Event):
    model: nn.Module = None
    optimizer: Any = None
    scheduler: Any = None
    runner: Runner = None


@dataclass
class ModelSaveEvent(Event):
    model: nn.Module = None
    optimizer: Any = None
    scheduler: Any = None
    epoch_index: int = None


@dataclass
class ModelSaveBestEvent(ModelSaveEvent):
    pass


@dataclass
class ModelSaveLastEvent(ModelSaveEvent):
    pass


@dataclass
class ModelForwardStartEvent(Event):
    input: torch.Tensor = None


@dataclass
class ModelForwardEndEvent(Event):
    output: torch.Tensor = None


@dataclass
class ModelLossStartEvent(Event):
    output: torch.Tensor = None
    target: torch.Tensor = None

    loader_name: str = None

    epoch_index: int = None
    step_index: int = None
    batch_index: int = None


@dataclass
class ModelLossEndEvent(Event):
    loss: nn.Module = None

    loader_name: str = None

    epoch_index: int = None
    step_index: int = None
    batch_index: int = None


@dataclass
class ModelBackwardStartEvent(Event):
    pass


@dataclass
class ModelBackwardEndEvent(Event):
    pass


@dataclass
class ModelScheduleStartEvent(Event):
    pass


@dataclass
class ModelScheduleEndEvent(Event):
    pass
