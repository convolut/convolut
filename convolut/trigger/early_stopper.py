from collections import deque

from decouple import Module

from ..constants import LoaderName
from ..events import RunnerForceStopEvent
from ..metric import MetricManagerFlushEvent


class EarlyStopper(Module):
    def __init__(self,
                 window: int = 3,
                 metric_name: str = "loss",
                 loader_name: str = LoaderName.Valid,
                 delta: float = 1e-4):
        super().__init__()

        self._window = window
        self._metric_name = metric_name
        self._loader_name = loader_name
        self._delta = delta

        self._values = deque(maxlen=self._window)
        self._last_epoch_index = -1

        (
            self.sub(MetricManagerFlushEvent, self.handle_metric_manager_flush)
        )

    def _early_stop(self):
        self.pub(RunnerForceStopEvent(reason="early_stopping"))

    def handle_metric_manager_flush(self, event: MetricManagerFlushEvent):
        metric_values = event.metrics[self._metric_name]

        for epoch_index, loaders in metric_values.items():
            for loader, value in loaders.items():
                if self._loader_name == loader:
                    if epoch_index > self._last_epoch_index:
                        self._last_epoch_index = epoch_index

                        if all([abs(v - value) < self._delta for v in self._values]) \
                                and len(self._values) == self._window:
                            self._early_stop()
                        else:
                            self._values.append(value)
