from typing import Dict, Any, Callable, Union, Tuple, Optional

import torch
from torch import nn
from decouple import Module
from .events import (
    ModelForwardStartEvent, ModelForwardEndEvent, ModelBackwardStartEvent, ModelBackwardEndEvent, ModelLossEndEvent,
    ModelLossStartEvent, ModelScheduleEndEvent, ModelScheduleStartEvent, ModelInitEvent,
    ModelSaveEvent)
from ..runner import RunnerStartEvent
from ..constants import LoaderName
from ..epoch import EpochStartEvent, EpochEndEvent
from ..loader import LoaderStartEvent, LoaderProcessBatchStartEvent


class ScheduleType:
    PerEpoch = "per_epoch"
    PerBatch = "per_batch"


class ModelManager(Module):
    def __init__(
            self,
            model: nn.Module,

            device,

            criterion,
            optimizer,
            scheduler,
            schedule_type: str = ScheduleType.PerEpoch,

            input_fn: Callable[[Any], torch.Tensor] = lambda batch: batch["input"],
            target_fn: Callable[[Any], torch.Tensor] = lambda batch: batch["target"],

            model_kwargs: Optional[Dict] = None,
            optim_fn: Union[Dict[int, Callable[[Any], Tuple[Any, Any]]], Callable[[Any], Tuple[Any, Any]]] = None,
    ):
        super().__init__()
        self._model = model
        self._model_kwargs = model_kwargs if model_kwargs else {}
        self._device = device

        self._criterion = criterion
        self._optimizer = optimizer

        self._scheduler = scheduler
        self._schedule_type = schedule_type

        self._input_fn = input_fn
        self._target_fn = target_fn

        self._optim_fn = optim_fn

        assert (self._optimizer and self._scheduler) or self._optim_fn

        self._current_epoch_index = None
        self._current_batch_index = None
        self._current_loader_name = None
        self._current_step = None
        self._current_backward_required = False

        self._current_output: torch.Tensor = None
        self._current_loss = None

        (
            self.sub(RunnerStartEvent, self.handle_runner_start)
                .sub(EpochStartEvent, self.handle_epoch_start)
                .sub(LoaderStartEvent, self.handle_loader_start)
                .sub(LoaderProcessBatchStartEvent, self.handle_process_batch_start)
                .sub(EpochEndEvent, self.handle_epoch_end)
        )

    def handle_runner_start(self, event: RunnerStartEvent):
        self.pub(ModelInitEvent(model=self._model,
                                optimizer=self._optimizer,
                                scheduler=self._scheduler,
                                runner=event.runner))

        self._model.to(self._device)

    def handle_epoch_start(self, event: EpochStartEvent):
        self._current_epoch_index = event.epoch.epoch_index

    def handle_loader_start(self, event: LoaderStartEvent):
        self._current_loader_name = event.loader.name
        self._current_backward_required = self._current_loader_name == LoaderName.Train

        if self._current_backward_required:
            self._model.train()
        else:
            self._model.eval()

    def handle_process_batch_start(self, event: LoaderProcessBatchStartEvent):
        self._current_loader_name = event.loader.name
        self._current_epoch_index = event.epoch_index
        self._current_batch_index = event.batch_index
        self._current_step = event.current_step

        inpt = self._input_fn(event.batch).to(self._device)
        target = self._target_fn(event.batch).to(self._device)

        if self._current_backward_required:
            self._forward(input=inpt)
            self._loss(output=self._current_output, target=target)
            self._backward()

            if self._schedule_type == ScheduleType.PerBatch:
                self._schedule()
        else:
            with torch.no_grad():
                self._forward(input=inpt)
                self._loss(output=self._current_output, target=target)

    def handle_epoch_end(self, event: EpochEndEvent):
        if self._schedule_type == ScheduleType.PerEpoch:
            self._schedule()

        self.pub(ModelSaveEvent(model=self._model,
                                optimizer=self._optimizer,
                                scheduler=self._scheduler,
                                epoch=event.epoch))

    def _forward(self, input: torch.Tensor):
        self.pub(ModelForwardStartEvent())

        output = self._model.forward(input, **self._model_kwargs)
        self._current_output = output

        self.pub(ModelForwardEndEvent())

    def _loss(self, output: torch.Tensor, target: torch.Tensor):
        self.pub(ModelLossStartEvent(output=output,
                                     target=target,
                                     loader_name=self._current_loader_name,
                                     step_index=self._current_step,
                                     batch_index=self._current_batch_index))

        loss = self._criterion(output, target)
        self._current_loss = loss

        self.pub(ModelLossEndEvent(loss=loss,
                                   epoch_index=self._current_epoch_index,
                                   loader_name=self._current_loader_name,
                                   step_index=self._current_step,
                                   batch_index=self._current_batch_index))

    def _backward(self):
        self.pub(ModelBackwardStartEvent())

        self._optimizer.zero_grad()
        self._current_loss.backward()
        self._optimizer.step()

        self.pub(ModelBackwardEndEvent())

    def _schedule(self):
        self.pub(ModelScheduleStartEvent())

        self._scheduler.step()

        self.pub(ModelScheduleEndEvent())
