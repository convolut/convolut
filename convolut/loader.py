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

        self._current_step = -1
        self._current_batch_index = 1

        self._loader_on: bool = False
        self._current_epoch_index = -1
        self._current_maximum_steps = None
        self._current_restart_unfinished = False
        self._debug = False

    def start(self,
              epoch_index: int,
              maximum_steps: Optional[int],
              restart_unfinished: Optional[bool],
              mediator: Mediator,
              debug: bool
              ):
        self.init(mediator)

        self._current_epoch_index = epoch_index
        self._current_maximum_steps = maximum_steps
        self._current_restart_unfinished = restart_unfinished
        self._debug = debug

        self.pub(LoaderStartEvent(loader=self,
                                  epoch_index=self._current_epoch_index,
                                  maximum_steps=self._current_maximum_steps,
                                  restart_unfinished=self._current_restart_unfinished,
                                  current_step=self._current_step,
                                  debug=self._debug))

        self._loader_on = True
        self._current_step = 1

        while self._loader_on and (
                (self._current_maximum_steps and self._current_step < self._current_maximum_steps)
                or self._current_maximum_steps is None):
            # create dataloader-iterator
            try:
                batch = next(self._iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                self._iterator = iter(self._dataloader)
                self._current_batch_index = 1
                batch = next(self._iterator)

            self.pub(LoaderProcessBatchStartEvent(loader=self,
                                                  batch=batch,
                                                  epoch_index=epoch_index,
                                                  batch_index=self._current_batch_index,
                                                  current_step=self._current_step,
                                                  debug=debug))

            self.pub(LoaderProcessBatchEndEvent(loader=self,
                                                epoch_index=epoch_index,
                                                batch_index=self._current_batch_index,
                                                current_step=self._current_step,
                                                debug=debug))

            self._current_step += 1
            self._current_batch_index += 1

        if self._current_restart_unfinished:
            self._iterator = iter(self._dataloader)
            self._current_batch_index = 1

        self.end()

    def handle_loader_force_stop(self, event: LoaderForceStopEvent):
        self.end()

    def end(self):
        self._loader_on = False
        self.pub(LoaderEndEvent(loader=self,
                                epoch_index=self._current_epoch_index,
                                maximum_steps=self._current_maximum_steps,
                                restart_unfinished=self._current_restart_unfinished,
                                current_step=self._current_step,
                                batch_index=self._current_batch_index,
                                debug=self._debug))


@dataclass
class LoaderStartEvent(Event):
    loader: Loader = None
    epoch_index: int = None
    maximum_steps: Optional[int] = None
    restart_unfinished: bool = None
    current_step: int = None
    debug: bool = None


@dataclass
class LoaderEndEvent(Event):
    loader: Loader = None
    epoch_index: int = None
    maximum_steps: Optional[int] = None
    restart_unfinished: bool = None
    current_step: int = None
    batch_index: int = None
    debug: bool = None


@dataclass
class LoaderProcessBatchStartEvent(Event):
    loader: Loader = None
    batch: Any = None
    epoch_index: int = None
    batch_index: int = None
    current_step: int = None
    debug: bool = None


@dataclass
class LoaderProcessBatchEndEvent(Event):
    loader: Loader = None
    epoch_index: int = None
    batch_index: int = None
    current_step: int = None
    debug: bool = None


class TrainLoader(Loader):
    def __init__(self, dataloader: Iterator):
        super().__init__(name=LoaderName.Train, dataloader=dataloader)


class ValidLoader(Loader):
    def __init__(self, dataloader: Iterator):
        super().__init__(name=LoaderName.Valid, dataloader=dataloader)
