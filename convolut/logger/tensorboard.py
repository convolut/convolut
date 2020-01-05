import os
from typing import Dict

from decouple import Module
from torch.utils.tensorboard import SummaryWriter

from ..constants import TensorboardMode
from ..runner import RunnerStartEvent, RunnerEndEvent
from ..epoch import EpochStartEvent, EpochEndEvent
from ..loader import LoaderStartEvent, LoaderEndEvent
from ..metric import MetricManagerFlushEvent


class TensorboardLogger(Module):
    def __init__(self,
                 folder: str,
                 mode: str = TensorboardMode.Basic):
        super().__init__()
        self._folder = folder
        self._mode = mode

        self._current_epoch_limit = 0
        self._current_epoch_index = 0
        self._current_loader_name = None
        self._last_loaders = {}
        self._writers: Dict[str, SummaryWriter] = {}

        (
            self.sub(RunnerStartEvent, self.handle_runner_start)
                .sub(EpochStartEvent, self.handle_epoch_start)
                .sub(LoaderStartEvent, self.handle_loader_start)
                .sub(LoaderEndEvent, self.handle_loader_end)
                .sub(EpochEndEvent, self.handle_epoch_end)
                .sub(RunnerEndEvent, self.handle_runner_end)
                .sub(MetricManagerFlushEvent, self.handle_metric_manager_flush)
        )

    def handle_runner_start(self, event: RunnerStartEvent):
        self._current_epoch_limit = event.runner.epochs_limit

    def handle_epoch_start(self, event: EpochStartEvent):
        self._current_epoch_index = event.epoch.epoch_index

    def handle_loader_start(self, event: LoaderStartEvent):
        self._current_loader_name = event.loader.name

        if self._current_loader_name not in self._writers:
            folder = os.path.join(self._folder, f'{self._current_loader_name}_log')
            self._writers[self._current_loader_name] = SummaryWriter(folder)

    def handle_loader_end(self, event: LoaderEndEvent):
        for writer in self._writers.values():
            writer.flush()

    def handle_epoch_end(self, event: EpochEndEvent):
        for writer in self._writers.values():
            writer.flush()

    def handle_runner_end(self, event: RunnerEndEvent):
        for writer in self._writers.values():
            writer.close()

    def handle_metric_manager_flush(self, event: MetricManagerFlushEvent):
        for metric, epochs in event.metrics.items():
            if metric not in self._last_loaders:
                self._last_loaders[metric] = {}

            for epoch, loaders in epochs.items():
                if epoch == self._current_epoch_index:
                    for loader, value in loaders.items():
                        self._last_loaders[metric][loader] = value

        for metric, loaders in self._last_loaders.items():
            for loader, value in loaders.items():
                self._writers[loader].add_scalar(f"{metric}/epoch", value, self._current_epoch_index)
