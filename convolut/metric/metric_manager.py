from dataclasses import dataclass
from statistics import mean
from typing import Dict, List

from decouple import Module, Event

from ..constants import FlushType
from .events import MetricEvent
from ..epoch import EpochEndEvent
from ..loader import LoaderEndEvent
from ..settings import METRIC_MANAGER_FLUSH_TYPE


class MetricManager(Module):
    def __init__(self,
                 flush_type: str = METRIC_MANAGER_FLUSH_TYPE
                 ):
        super().__init__()

        self._flush_type = flush_type
        self._raw: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
        self._metrics: Dict[str, Dict[int, Dict[str, float]]] = {}

        self.sub(MetricEvent, self.handle_metric)

        if self._flush_type == FlushType.PerEpoch:
            self.sub(EpochEndEvent, self._flush_per_epoch)
        elif self._flush_type == FlushType.PerLoader:
            self.sub(LoaderEndEvent, self._flush_per_loader)
        else:
            raise ValueError(f'unknown flush_type:{self._flush_type}')

    def _flush_per_epoch(self, event: EpochEndEvent):
        epoch_index = event.epoch.epoch_index

        metrics = self._aggregate(epoch_index)

        self.pub(MetricManagerFlushEvent(metrics=metrics))

    def _flush_per_loader(self, event: LoaderEndEvent):
        epoch_index = event.epoch_index

        metrics = self._aggregate(epoch_index)

        self.pub(MetricManagerFlushEvent(metrics=metrics))

    def _aggregate(self, epoch: int) -> Dict[str, Dict[int, Dict[str, float]]]:
        for metric in self._raw.keys():
            if metric not in self._metrics:
                self._metrics[metric] = {}

            for epoch, loaders in self._raw[metric].items():
                if epoch not in self._metrics[metric]:
                    self._metrics[metric][epoch] = {}

                for loader, values in loaders.items():
                    if loader not in self._metrics[metric][epoch]:
                        self._metrics[metric][epoch][loader] = mean(values)

        return self._metrics

    def handle_metric(self, event: MetricEvent):
        metric_name = event.metric_name
        metric_value = event.metric_value

        epoch_index = event.periods["epoch_index"]
        loader_name = event.periods["loader_name"]
        step_index = event.periods["step_index"]  # todo
        batch_index = event.periods["batch_index"]  # todo

        if metric_name not in self._raw:
            self._raw[metric_name] = {}

        if epoch_index not in self._raw[metric_name]:
            self._raw[metric_name][epoch_index] = {}

        if loader_name not in self._raw[metric_name][epoch_index]:
            self._raw[metric_name][epoch_index][loader_name] = []

        self._raw[metric_name][epoch_index][loader_name].append(metric_value)


@dataclass
class MetricManagerFlushEvent(Event):
    metrics: Dict[str, Dict[int, Dict[str, float]]] = None  # metric_name->epoch_index->loader_name,value
