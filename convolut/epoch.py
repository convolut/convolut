from dataclasses import dataclass
from typing import List, Optional

from decouple import Mediator, Event, Module

from .loader import Loader
from .events import EpochForceStopEvent, LoaderForceStopEvent


class Epoch(Module):
    def __init__(self,
                 epoch_index: int,

                 loaders: List[Loader],
                 steps_per_epoch: Optional[int],
                 restart_iterator: Optional[bool],

                 mediator: Mediator,
                 ):
        super().__init__(mediator)
        self.epoch_index = epoch_index

        self._loaders = loaders
        self._steps_per_epoch = steps_per_epoch
        self._restart_iterator = restart_iterator

        self._epoch_on = False
        self._current_loader_name = None
        self._current_loader: Loader = None

        (
            self.sub(EpochForceStopEvent, self.handle_epoch_force_stop)
        )

    def start(self):
        self.pub(EpochStartEvent(epoch=self))

        self._epoch_on = True
        for loader in self._loaders:
            if not self._epoch_on:
                break

            self._current_loader_name = loader.name
            self._current_loader = loader

            self.preprocess_loader(loader=loader)
            self.process_loader(loader=loader)
            self.postprocess_loader(loader=loader)

        self.end()

    def handle_epoch_force_stop(self, event: EpochForceStopEvent):
        self.pub(LoaderForceStopEvent())
        self.end()

    def end(self):
        self._epoch_on = False
        self.pub(EpochEndEvent(epoch=self))

    def preprocess_loader(self, loader: Loader):
        self.pub(EpochPreprocessLoaderEvent(epoch=self, loader=loader))

    def process_loader(self, loader: Loader):
        self.pub(EpochProcessLoaderEvent(epoch=self, loader=loader))

        loader.start(epoch_index=self.epoch_index,
                     maximum_steps=self._steps_per_epoch,
                     restart_iterator=self._restart_iterator,
                     mediator=self._mediator)

    def postprocess_loader(self, loader: Loader):
        self.pub(EpochPostprocessLoaderEvent(epoch=self, loader=loader))


@dataclass
class EpochStartEvent(Event):
    epoch: Epoch = None


@dataclass
class EpochEndEvent(Event):
    epoch: Epoch = None


@dataclass
class EpochPreprocessLoaderEvent(Event):
    epoch: Epoch = None
    loader: Loader = None


@dataclass
class EpochProcessLoaderEvent(Event):
    epoch: Epoch = None
    loader: Loader = None


@dataclass
class EpochPostprocessLoaderEvent(Event):
    epoch: Epoch = None
    loader: Loader = None
