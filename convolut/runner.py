from dataclasses import dataclass
from typing import List, Dict, Union, Optional

from decouple import Event, Mediator, Module

from .epoch import Epoch
from .events import RunnerForceStopEvent, EpochForceStopEvent
from .loader import Loader
from .settings import (
    RUNNER_EPOCHS,
    RUNNER_STEPS_PER_EPOCH,
    RUNNER_RESTART_ITERATOR
)


class Runner(Module):
    def __init__(self,
                 loaders: Union[Dict[int, List[Loader]], List[Loader]],

                 epochs: int = RUNNER_EPOCHS,
                 steps_per_epoch: Optional[int] = RUNNER_STEPS_PER_EPOCH,
                 restart_iterator: Optional[bool] = RUNNER_RESTART_ITERATOR,
                 mediator: Mediator = Mediator()
                 ):
        super().__init__(mediator)

        self._loaders = loaders
        self._restart_iterator = restart_iterator

        epochs_limit = 0
        if isinstance(self._loaders, dict):
            epochs_limit = max([epochs_limit] + list(self._loaders.keys()))

        self.epochs_limit = epochs if epochs >= epochs_limit else epochs_limit

        self._epochs: List[Epoch] = []
        self._steps_per_epoch = steps_per_epoch

        self._runner_on = False
        self.current_epoch_index = 0
        self._current_epoch: Optional[Epoch] = None

        (
            self.sub(RunnerForceStopEvent, self.handle_runner_force_stop)
        )

    def _create_epoch(self, epoch_index: int) -> Epoch:
        if isinstance(self._loaders, dict):
            idx = [i for i in sorted(list(self._loaders.keys())) if i <= epoch_index][-1]
            loaders = self._loaders[idx]
        else:
            loaders = self._loaders

        epoch = Epoch(epoch_index=epoch_index,
                      loaders=loaders,
                      steps_per_epoch=self._steps_per_epoch,
                      restart_iterator=self._restart_iterator,
                      mediator=self._mediator)

        return epoch

    def start(self):
        self.pub(RunnerStartEvent(runner=self))

        self._runner_on = True

        while self._runner_on and self.current_epoch_index < self.epochs_limit:
            self.current_epoch_index += 1
            epoch = self._create_epoch(epoch_index=self.current_epoch_index)
            self._epochs.append(epoch)
            self._current_epoch = epoch

            self.preprocess_epoch(epoch=epoch)
            self.process_epoch(epoch=epoch)
            self.postprocess_epoch(epoch=epoch)

        self.end()

    def handle_runner_force_stop(self, event: RunnerForceStopEvent):
        self.pub(EpochForceStopEvent())
        self.end()

    def end(self):
        self._runner_on = False
        self.pub(RunnerEndEvent(runner=self))

    def preprocess_epoch(self, epoch: Epoch):
        self.pub(RunnerPreprocessEpochEvent(runner=self, epoch=epoch))

    def process_epoch(self, epoch: Epoch):
        self.pub(RunnerProcessEpochEvent(runner=self, epoch=epoch))

        epoch.start()

    def postprocess_epoch(self, epoch: Epoch):
        self.pub(RunnerPostprocessEpochEvent(runner=self, epoch=epoch))


@dataclass
class RunnerStartEvent(Event):
    runner: Runner = None


@dataclass
class RunnerEndEvent(Event):
    runner: Runner = None


@dataclass
class RunnerPreprocessEpochEvent(Event):
    runner: Runner = None
    epoch: Epoch = None


@dataclass
class RunnerProcessEpochEvent(Event):
    runner: Runner = None
    epoch: Epoch = None


@dataclass
class RunnerPostprocessEpochEvent(Event):
    runner: Runner = None
    epoch: Epoch = None
