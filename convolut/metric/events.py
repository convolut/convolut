from dataclasses import dataclass
from typing import Dict, Union

from decouple import Event


@dataclass
class MetricEvent(Event):
    metric_name: str = None
    metric_value: float = None

    periods: Dict[str, Union[str, int]] = None
