from dataclasses import dataclass
from typing import List, Dict, Union, Optional

from decouple import Event, Mediator, Module

from .epoch import Epoch
from .events import RunnerForceStopEvent, EpochForceStopEvent
from .loader import Loader


class Runner(Module):
    def __init__(self,
                 loaders: Union[Dict[int, List[Loader]], List[Loader]],

                 epochs: int = 1,
                 steps_per_epoch: Optional[int] = None,
                 restart_unfinished_loader: Optional[bool] = None,
                 mediator: Mediator = Mediator(),
                 debug: bool = False,
                 ):
        super().__init__(mediator)

        self._loaders = loaders
        self._restart_unfinished_loader = restart_unfinished_loader

        epochs_limit = 0
        if isinstance(self._loaders, dict):
            epochs_limit = max([epochs_limit] + list(self._loaders.keys()))

        self.epochs_limit = epochs if epochs >= epochs_limit else epochs_limit

        self._epochs: List[Epoch] = []
        self._steps_per_epoch = steps_per_epoch
        self._debug = debug

        self._runner_on = False
        self._current_epoch_index = -1
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
                      restart_unfinished_loader=self._restart_unfinished_loader,
                      mediator=self._mediator,
                      debug=self._debug)

        return epoch

    def start(self):
        self.pub(RunnerStartEvent(runner=self))

        self._runner_on = True
        self._current_epoch_index = 1

        while self._runner_on and self._current_epoch_index <= self.epochs_limit:
            epoch = self._create_epoch(epoch_index=self._current_epoch_index)
            self._epochs.append(epoch)
            self._current_epoch = epoch

            self.preprocess_epoch(epoch=epoch)
            self.process_epoch(epoch=epoch)
            self.postprocess_epoch(epoch=epoch)

            self._current_epoch_index += 1

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
