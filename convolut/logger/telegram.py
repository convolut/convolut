from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from decouple import Module

from ..epoch import EpochStartEvent
from ..loader import LoaderStartEvent
from ..metric import MetricManagerFlushEvent
from ..runner import RunnerStartEvent, RunnerEndEvent, RunnerForceStopEvent
from ..settings import (
    LOGGER_TELEGRAM_TOKEN,
    LOGGER_TELEGRAM_CHANNEL,
    LOGGER_TELEGRAM_MODE,
    LOGGER_TELEGRAM_PROXY
)


class TelegramLogger(Module):
    def __init__(self,
                 token: str = LOGGER_TELEGRAM_TOKEN,
                 channel: str = LOGGER_TELEGRAM_CHANNEL,
                 mode: str = LOGGER_TELEGRAM_MODE,
                 proxy: str = LOGGER_TELEGRAM_PROXY,
                 ):
        super().__init__()
        self._token = token
        self._channel = channel
        self._mode = mode
        self._proxy = proxy
        self._base_url = f"{self._proxy}/bot{self._token}/sendMessage"

        self._current_epochs_limit = 0
        self._current_epoch_index = 0
        self._current_loader_name = None
        self._last_loaders = {}

        (
            self.sub(RunnerStartEvent, self.handle_runner_start)
                .sub(EpochStartEvent, self.handle_epoch_start)
                .sub(LoaderStartEvent, self.handle_loader_start)
                .sub(RunnerForceStopEvent, self.handle_runner_force_stop)
                .sub(RunnerEndEvent, self.handle_runner_end)
                .sub(MetricManagerFlushEvent, self.handle_metric_manager_flush)
        )

    def _send(self, text: str):
        try:
            quoted_text = quote_plus(text, safe='')
            url = f"{self._base_url}?chat_id={self._channel}&disable_web_page_preview=1&text={quoted_text}"

            request = Request(url)
            urlopen(request)
        except Exception as e:
            print(e)

    def handle_runner_start(self, event: RunnerStartEvent):
        self._current_epochs_limit = event.runner.epochs_limit

        text = f"runner.start"
        self._send(text)

    def handle_epoch_start(self, event: EpochStartEvent):
        self._current_epoch_index = event.epoch.epoch_index

    def handle_loader_start(self, event: LoaderStartEvent):
        self._current_loader_name = event.loader.name

    def handle_runner_force_stop(self, event: RunnerForceStopEvent):
        text = f"runner.force_stop.reason={event.reason}"
        self._send(text)

    def handle_runner_end(self, event: RunnerEndEvent):
        text = f"runner.end"
        self._send(text)

    def handle_metric_manager_flush(self, event: MetricManagerFlushEvent):
        text = f"epoch {self._current_epoch_index}/{self._current_epochs_limit} "

        for metric, epochs in event.metrics.items():
            if metric not in self._last_loaders:
                self._last_loaders[metric] = {}

            for epoch, loaders in epochs.items():
                if epoch == self._current_epoch_index:
                    for loader, value in loaders.items():
                        self._last_loaders[metric][loader] = value

        for metric, loaders in self._last_loaders.items():
            for loader, value in loaders.items():
                if text:
                    text = text + f", ({loader}){metric}: {value:3.4f}"
                else:
                    text = text + f"({loader}){metric}: {value:3.4f}"

        self._send(text)
