from dataclasses import dataclass

from decouple import Event


@dataclass
class RunnerForceStopEvent(Event):
    reason: str = None


@dataclass
class EpochForceStopEvent(Event):
    pass


@dataclass
class LoaderForceStopEvent(Event):
    pass
