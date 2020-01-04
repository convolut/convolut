from decouple import Module

from .events import MetricEvent
from ..model import ModelLossStartEvent
from ..utils.dice import dice


class DiceMetric(Module):
    def __init__(self):
        super().__init__()

        self.sub(ModelLossStartEvent, self.dice)

    def dice(self, event: ModelLossStartEvent):
        metric_value = dice(event.output, event.target)

        self.pub(MetricEvent(metric_name="dice",
                             metric_value=metric_value,
                             periods={
                                 "loader_name": event.loader_name,
                                 "epoch_index": event.epoch_index,
                                 "step_index": event.step_index,
                                 "batch_index": event.batch_index,
                             }))
