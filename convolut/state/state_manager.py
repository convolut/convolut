from decouple import Module

from .events import (
    CheckpointSavingEvent,
    CheckpointSavedEvent,
    CheckpointLoadingEvent,
    CheckpointLoadedEvent,
    StateSaveEvent,
    StateLoadEvent
)
from ..model import ModelInitEvent, ModelSaveEvent


class StateManager(Module):
    def __init__(self):
        super().__init__()

        self._model = None
        self._optimizer = None
        self._scheduler = None

        (
            self.sub(CheckpointSavingEvent, self.handle_checkpoint_saving)
                .sub(CheckpointSavedEvent, self.handle_checkpoint_saved)
                .sub(CheckpointLoadingEvent, self.handle_checkpoint_loading)
                .sub(CheckpointLoadedEvent, self.handle_checkpoint_loaded)
                .sub(ModelInitEvent, self.load_last_state)
                .sub(ModelSaveEvent, self.save_last_state)
        )

    def handle_checkpoint_saving(self, event: CheckpointSavingEvent):
        print(event)

    def handle_checkpoint_saved(self, event: CheckpointSavedEvent):
        print(event)

    def handle_checkpoint_loading(self, event: CheckpointLoadingEvent):
        print(event)

    def handle_checkpoint_loaded(self, event: CheckpointLoadedEvent):
        model_state = event.checkpoint["model"]
        optimizer_state = event.checkpoint["optimizer"]
        scheduler_state = event.checkpoint["scheduler"]
        epoch_index = event.checkpoint["epoch_index"]

        state_type = event.checkpoint_type

        print(event)

    def save_last_state(self, event: ModelSaveEvent):
        state = {
            "model": event.model,
            "optimizer": event.optimizer,
            "scheduler": event.scheduler,
            "epoch_index": event.epoch.epoch_index
        }
        state_type = "last"

        self.pub(StateSaveEvent(state=state, state_type=state_type))

    def load_last_state(self, event: ModelInitEvent):
        self._model = event.model
        self._optimizer = event.optimizer
        self._scheduler = event.scheduler
        state_type = "last"

        self.pub(StateLoadEvent(state_type=state_type))
