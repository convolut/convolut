from typing import Dict, Any, Callable, Optional

import torch
from decouple import Module
from torch import nn

from .events import (
    ModelForwardStartEvent, ModelForwardEndEvent, ModelInitEvent)
from ..loader import LoaderProcessBatchStartEvent
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

        (
            self.sub(RunnerStartEvent, self.handle_runner_start)
                .sub(LoaderProcessBatchStartEvent, self.handle_process_batch_start)
        )

    def handle_runner_start(self, event: RunnerStartEvent):
        self.pub(ModelInitEvent(model=self._model,
                                runner=event.runner))

        self._model.eval()
        self._model.to(self._device)

    def handle_process_batch_start(self, event: LoaderProcessBatchStartEvent):
        inpt = self._input_fn(event.batch).to(self._device)
        with torch.no_grad():
            self._forward(input=inpt)

    def _forward(self, input: torch.Tensor):
        self.pub(ModelForwardStartEvent(input=input))

        output = self._model.forward(input, **self._model_kwargs)

        self.pub(ModelForwardEndEvent(output=output))
