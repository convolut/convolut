from decouple import Module

from .events import MetricEvent
from ..model import ModelLossEndEvent


class LossMetric(Module):
    def __init__(self):
        super().__init__()

        self.sub(ModelLossEndEvent, self.loss)

    def loss(self, event: ModelLossEndEvent):
        metric_value = event.loss.item()

        self.pub(MetricEvent(metric_name="loss",
                             metric_value=metric_value,
                             periods={
                                 "loader_name": event.loader_name,
                                 "epoch_index": event.epoch_index,
                                 "step_index": event.step_index,
                                 "batch_index": event.batch_index,
                             }))
