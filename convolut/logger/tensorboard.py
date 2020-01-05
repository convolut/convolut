import os
from typing import Dict

from decouple import Module
from torch.utils.tensorboard import SummaryWriter

from ..epoch import EpochStartEvent, EpochEndEvent
from ..loader import LoaderStartEvent, LoaderEndEvent, LoaderProcessBatchEndEvent
from ..metric import MetricManagerFlushEvent, MetricEvent
from ..runner import RunnerEndEvent
from ..settings import (
    LOGGER_TENSORBOARD_FOLDER,
    LOGGER_TENSORBOARD_MODE
)


class TensorboardLogger(Module):
    def __init__(self,
                 folder: str = LOGGER_TENSORBOARD_FOLDER,
                 mode: str = LOGGER_TENSORBOARD_MODE,
                 ):
        super().__init__()
        self._folder = folder
        os.makedirs(self._folder, exist_ok=True)

        self._mode = mode

        self._current_epoch_index = -1

        self._current_global_steps: Dict[str, int] = {}
        self._current_loader_name = None

        self._last_loaders = {}
        self._writers: Dict[str, SummaryWriter] = {}

        (
            self.sub(EpochStartEvent, self.handle_epoch_start)
                .sub(LoaderStartEvent, self.handle_loader_start)
                .sub(LoaderProcessBatchEndEvent, self.handle_loader_process_batch_end)
                .sub(LoaderEndEvent, self.handle_loader_end)
                .sub(EpochEndEvent, self.handle_epoch_end)
                .sub(RunnerEndEvent, self.handle_runner_end)
                .sub(MetricEvent, self.handle_metric)
                .sub(MetricManagerFlushEvent, self.handle_metric_manager_flush)
        )

    def handle_epoch_start(self, event: EpochStartEvent):
        self._current_epoch_index = event.epoch.epoch_index

    def _lazy_init_writer(self, loader_name: str):
        if loader_name not in self._writers:
            folder = os.path.join(self._folder, f'{loader_name}')
            self._writers[loader_name] = SummaryWriter(folder, )

        if loader_name not in self._current_global_steps:
            self._current_global_steps[loader_name] = 0

    def handle_loader_start(self, event: LoaderStartEvent):
        self._current_loader_name = event.loader.name

        self._lazy_init_writer(self._current_loader_name)

    def handle_loader_process_batch_end(self, event: LoaderProcessBatchEndEvent):
        loader_name = event.loader.name
        self._lazy_init_writer(loader_name)

        self._current_global_steps[loader_name] += 1

    def handle_loader_end(self, event: LoaderEndEvent):
        for writer in self._writers.values():
            writer.flush()

    def handle_epoch_end(self, event: EpochEndEvent):
        for writer in self._writers.values():
            writer.flush()

    def handle_runner_end(self, event: RunnerEndEvent):
        for writer in self._writers.values():
            writer.close()

    def handle_metric(self, event: MetricEvent):
        metric_name = event.metric_name
        metric_value = event.metric_value

        loader_name = event.periods["loader_name"]
        self._lazy_init_writer(loader_name)

        tag = f"{metric_name}/step"

        self._writers[loader_name].add_scalar(tag=tag,
                                              scalar_value=metric_value,
                                              global_step=self._current_global_steps[loader_name])

    def handle_metric_manager_flush(self, event: MetricManagerFlushEvent):
        # get only last epoch's loaders metrics
        for metric, epochs in event.metrics.items():
            if metric not in self._last_loaders:
                self._last_loaders[metric] = {}

            for epoch, loaders in epochs.items():
                if epoch == self._current_epoch_index:
                    for loader, value in loaders.items():
                        self._last_loaders[metric][loader] = value

        for metric, loaders in self._last_loaders.items():
            for loader, value in loaders.items():
                tag = f"{metric}/epoch"
                self._writers[loader].add_scalar(tag=tag,
                                                 scalar_value=value,
                                                 global_step=self._current_epoch_index)
