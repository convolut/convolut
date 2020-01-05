from typing import Dict, Any, Callable, Optional

import torch
from decouple import Module
from torch import nn

from .events import (
    ModelForwardStartEvent, ModelForwardEndEvent, ModelInitEvent)
from ..epoch import EpochStartEvent, EpochEndEvent
from ..loader import LoaderStartEvent, LoaderProcessBatchStartEvent
from ..runner import RunnerStartEvent


class InferManager(Module):
    def __init__(
            self,
            model: nn.Module,
            device,
            input_fn: Callable[[Any], torch.Tensor] = lambda batch: batch["input"],
            model_kwargs: Optional[Dict] = None
    ):
        super().__init__()
        self._model = model
        self._model_kwargs = model_kwargs if model_kwargs else {}
        self._device = device

        self._input_fn = input_fn

        self._current_epoch_index = None
        self._current_step = None
        self._current_batch_index = None

        self._current_loader_name = None

        self._current_output: torch.Tensor = None

        (
            self.sub(RunnerStartEvent, self.handle_runner_start)
                .sub(EpochStartEvent, self.handle_epoch_start)
                .sub(LoaderStartEvent, self.handle_loader_start)
                .sub(LoaderProcessBatchStartEvent, self.handle_process_batch_start)
                .sub(EpochEndEvent, self.handle_epoch_end)
        )

    def handle_runner_start(self, event: RunnerStartEvent):
        self.pub(ModelInitEvent(model=self._model,
                                runner=event.runner))

        self._model.to(self._device)

    def handle_epoch_start(self, event: EpochStartEvent):
        self._current_epoch_index = event.epoch.epoch_index

    def handle_loader_start(self, event: LoaderStartEvent):
        self._current_loader_name = event.loader.name

        self._model.eval()

    def handle_process_batch_start(self, event: LoaderProcessBatchStartEvent):
        self._current_loader_name = event.loader.name
        self._current_epoch_index = event.epoch_index
        self._current_step = event.current_step
        self._current_batch_index = event.batch_index

        inpt = self._input_fn(event.batch).to(self._device)
        with torch.no_grad():
            self._forward(input=inpt)

    def handle_epoch_end(self, event: EpochEndEvent):
        pass

    def _forward(self, input: torch.Tensor):
        self.pub(ModelForwardStartEvent())

        output = self._model.forward(input, **self._model_kwargs)
        self._current_output = output

        self.pub(ModelForwardEndEvent())
