import torch
from decouple import Module

from .events import (
    CheckpointSavingEvent,
    CheckpointSavedEvent,
    CheckpointLoadingEvent,
    CheckpointLoadedEvent,
    StateSaveEvent,
    StateLoadEvent
)
from ..constants import StateMode
from ..model import ModelInitEvent, ModelSaveLastEvent, ModelSaveBestEvent
from ..runner import Runner
from ..settings import STATE_MANAGER_STATE_MODE


class StateManager(Module):
    def __init__(self,
                 mode: str = STATE_MANAGER_STATE_MODE,
                 ):
        super().__init__()

        self._mode = mode

        self._runner: Runner = None
        self._model: torch.nn.Module = None
        self._optimizer = None
        self._scheduler = None

        (
            self.sub(CheckpointSavingEvent, self.handle_checkpoint_saving)
                .sub(CheckpointSavedEvent, self.handle_checkpoint_saved)
                .sub(CheckpointLoadingEvent, self.handle_checkpoint_loading)
                .sub(CheckpointLoadedEvent, self.handle_checkpoint_loaded)
                .sub(ModelInitEvent, self.handle_model_init)
                .sub(ModelSaveLastEvent, self.handle_model_save_last)
                .sub(ModelSaveBestEvent, self.handle_model_save_best)
        )

    def handle_checkpoint_saving(self, event: CheckpointSavingEvent):
        pass

    def handle_checkpoint_saved(self, event: CheckpointSavedEvent):
        pass

    def handle_checkpoint_loading(self, event: CheckpointLoadingEvent):
        pass

    def handle_checkpoint_loaded(self, event: CheckpointLoadedEvent):
        model_state_dict = event.checkpoint["model_state_dict"]
        self._model.load_state_dict(model_state_dict)

        optimizer_state_dict = event.checkpoint["optimizer_state_dict"]
        self._optimizer.load_state_dict(optimizer_state_dict)

        scheduler_state_dict = event.checkpoint["scheduler_state_dict"]
        self._scheduler.load_state_dict(scheduler_state_dict)

        epoch_index = event.checkpoint["epoch_index"]
        self._runner.current_epoch_index = epoch_index

    def handle_model_save_last(self, event: ModelSaveLastEvent):
        state = {
            "model_state_dict": event.model.state_dict(),
            "optimizer_state_dict": event.optimizer.state_dict(),
            "scheduler_state_dict": event.scheduler.state_dict(),
            "epoch_index": event.epoch_index
        }

        self.pub(StateSaveEvent(state=state, state_type=StateMode.Last))

    def handle_model_save_best(self, event: ModelSaveBestEvent):
        state = {
            "model_state_dict": event.model.state_dict(),
            "optimizer_state_dict": event.optimizer.state_dict(),
            "scheduler_state_dict": event.scheduler.state_dict(),
            "epoch_index": event.epoch_index
        }

        self.pub(StateSaveEvent(state=state, state_type=StateMode.Best))

    def handle_model_init(self, event: ModelInitEvent):
        self._runner = event.runner
        self._model = event.model
        self._optimizer = event.optimizer
        self._scheduler = event.scheduler

        self.pub(StateLoadEvent(state_type=self._mode))
