from dataclasses import dataclass
from typing import Iterator, Optional, Any

from decouple import Module, Mediator, Event
from .constants.loader_name import LoaderName
from .events import LoaderForceStopEvent


class Loader(Module):
    def __init__(self, name: str, dataloader: Iterator):
        super().__init__()

        self.name = name
        self._dataloader = dataloader
        self._iterator = iter(self._dataloader)

        self._loader_on: bool = False

        self._current_epoch_index = -1
        self._current_step_index = -1
        self._current_batch_index = 0
        self._current_maximum_steps = None
        self._current_restart_iterator = False

    def start(self,
              epoch_index: int,
              maximum_steps: Optional[int],
              restart_iterator: Optional[bool],
              mediator: Mediator,
              ):
        self.init(mediator)

        self._current_epoch_index = epoch_index
        self._current_maximum_steps = maximum_steps
        self._current_restart_iterator = restart_iterator

        self.pub(LoaderStartEvent(loader=self,
                                  epoch_index=self._current_epoch_index,
                                  maximum_steps=self._current_maximum_steps,
                                  restart_iterator=self._current_restart_iterator,
                                  step_index=self._current_step_index))

        self._loader_on = True
        self._current_step_index = 0

        while self._loader_on and (
                (self._current_maximum_steps and self._current_step_index < self._current_maximum_steps)
                or self._current_maximum_steps is None):
            # create dataloader's iterator
            try:
                batch = next(self._iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                self._restart_iterator()
                batch = next(self._iterator)

            self.pub(LoaderProcessBatchStartEvent(loader=self,
                                                  batch=batch,
                                                  epoch_index=epoch_index,
                                                  step_index=self._current_step_index,
                                                  batch_index=self._current_batch_index))

            self.pub(LoaderProcessBatchEndEvent(loader=self,
                                                epoch_index=epoch_index,
                                                step_index=self._current_step_index,
                                                batch_index=self._current_batch_index))

            self._current_step_index += 1
            self._current_batch_index += 1

        if self._current_restart_iterator:
            self._restart_iterator()

        self.end()

    def _restart_iterator(self):
        self._iterator = iter(self._dataloader)
        self._current_batch_index = 0

    def handle_loader_force_stop(self, event: LoaderForceStopEvent):
        self.end()

    def end(self):
        self._loader_on = False
        self.pub(LoaderEndEvent(loader=self,
                                epoch_index=self._current_epoch_index,
                                step_index=self._current_step_index,
                                batch_index=self._current_batch_index,
                                maximum_steps=self._current_maximum_steps,
                                restart_iterator=self._current_restart_iterator))


@dataclass
class LoaderStartEvent(Event):
    loader: Loader = None
    epoch_index: int = None
    step_index: int = None
    maximum_steps: Optional[int] = None
    restart_iterator: bool = None


@dataclass
class LoaderEndEvent(Event):
    loader: Loader = None
    epoch_index: int = None
    step_index: int = None
    batch_index: int = None
    maximum_steps: Optional[int] = None
    restart_iterator: bool = None


@dataclass
class LoaderProcessBatchStartEvent(Event):
    loader: Loader = None
    epoch_index: int = None
    step_index: int = None
    batch_index: int = None
    batch: Any = None


@dataclass
class LoaderProcessBatchEndEvent(Event):
    loader: Loader = None
    epoch_index: int = None
    step_index: int = None
    batch_index: int = None


class TrainLoader(Loader):
    def __init__(self, dataloader: Iterator):
        super().__init__(name=LoaderName.Train, dataloader=dataloader)


class ValidLoader(Loader):
    def __init__(self, dataloader: Iterator):
        super().__init__(name=LoaderName.Valid, dataloader=dataloader)


class InferLoader(Loader):
    def __init__(self, dataloader: Iterator):
        super().__init__(name=LoaderName.Infer, dataloader=dataloader)
