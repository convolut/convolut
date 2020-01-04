from dataclasses import dataclass
from typing import Any, Dict

from decouple import Event


@dataclass
class StateSaveEvent(Event):
    state: Dict[str, Any] = None
    state_type: str = None


@dataclass
class StateLoadEvent(Event):
    state_type: str = None


@dataclass
class CheckpointLoadingEvent(Event):
    checkpoint_type: str = None


@dataclass
class CheckpointLoadedEvent(Event):
    checkpoint: Dict[str, Any] = None
    checkpoint_type: str = None


@dataclass
class CheckpointSavingEvent(Event):
    checkpoint: Dict[str, Any] = None
    checkpoint_type: str = None


@dataclass
class CheckpointSavedEvent(Event):
    checkpoint_type: str = None
