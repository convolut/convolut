from typing import Any, Dict
import os
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
from ..settings import STATE_FILE_CHECKPOINT_FOLDER, STATE_FILE_CHECKPOINT_SUFFIX


class FileCheckpoint(Module):
    def __init__(
            self,
            folder: str = STATE_FILE_CHECKPOINT_FOLDER,
            suffix: str = STATE_FILE_CHECKPOINT_SUFFIX
    ):
        super().__init__()
        self._folder = folder
        os.makedirs(self._folder, exist_ok=True)

        self._suffix = suffix
        self._filenames = {
            StateMode.Best: f'{StateMode.Best}{self._suffix}',
            StateMode.Last: f'{StateMode.Last}{self._suffix}',
        }

        self._filepaths = {}

        for key, filename in self._filenames.items():
            self._filepaths[key] = os.path.join(self._folder, filename)

        (
            self.sub(StateSaveEvent, self.handle_state_save)
                .sub(StateLoadEvent, self.handle_state_load)
        )

    def handle_state_save(self, event: StateSaveEvent):
        filepath = self._checkpoint_path(event.state_type)

        self._save_checkpoint(checkpoint=event.state,
                              checkpoint_type=event.state_type,
                              filepath=filepath)

    def handle_state_load(self, event: StateLoadEvent):
        filepath = self._checkpoint_path(event.state_type)

        if os.path.exists(filepath):
            self._load_checkpoint(checkpoint_type=event.state_type,
                                  filepath=filepath)

    def _checkpoint_path(self, checkpoint_type: str) -> str:
        if checkpoint_type not in self._filenames:
            self._filenames[checkpoint_type] = f'{checkpoint_type}_{self._suffix}'
            self._filepaths[checkpoint_type] = os.path.join(self._folder, self._filenames[checkpoint_type])

        filepath = self._filepaths[checkpoint_type]

        return filepath

    def _save_checkpoint(self, checkpoint: Dict[str, Any], checkpoint_type: str, filepath: str):
        self.pub(CheckpointSavingEvent(checkpoint=checkpoint, checkpoint_type=checkpoint_type))

        torch.save(checkpoint, filepath)

        self.pub(CheckpointSavedEvent(checkpoint_type=checkpoint_type))

    def _load_checkpoint(self, checkpoint_type: str, filepath: str):
        self.pub(CheckpointLoadingEvent(checkpoint_type=checkpoint_type))

        checkpoint = torch.load(filepath)

        self.pub(CheckpointLoadedEvent(checkpoint=checkpoint, checkpoint_type=checkpoint_type))
