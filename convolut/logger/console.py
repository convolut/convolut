import sys
from typing import Dict

from decouple import Module

from ..constants import AnsiColor as Color, ConsoleMode, LoaderName
from ..epoch import EpochStartEvent
from ..events import RunnerForceStopEvent
from ..loader import LoaderStartEvent, LoaderProcessBatchEndEvent
from ..metric.metric_manager import MetricManagerFlushEvent
from ..state.events import CheckpointSavedEvent, CheckpointSavingEvent, CheckpointLoadedEvent, CheckpointLoadingEvent
from ..runner import RunnerStartEvent
from ..settings import LOGGER_CONSOLE_MODE


class ConsoleLogger(Module):
    def __init__(self,
                 mode: str = LOGGER_CONSOLE_MODE,
                 ):
        super().__init__()

        self._mode = mode

        self._current_epochs_limit = 0
        self._current_epoch_index = 0

        self._current_global_steps: Dict[str, int] = {}  # todo

        self._current_loader_name = None
        self._current_text = ''
        self._info = ''
        self._last_loaders = {}

        (
            self.sub(RunnerStartEvent, self.handle_runner_start)
                .sub(RunnerForceStopEvent, self.handle_runner_force_stop)
                .sub(EpochStartEvent, self.handle_epoch_start)
                .sub(LoaderStartEvent, self.handle_loader_start)
                .sub(LoaderProcessBatchEndEvent, self.handle_loader_process_batch_end)
                .sub(MetricManagerFlushEvent, self.handle_metric_manager_flush)
                .sub(CheckpointSavingEvent, self.handle_checkpoint_saving)
                .sub(CheckpointSavedEvent, self.handle_checkpoint_saved)
                .sub(CheckpointLoadingEvent, self.handle_checkpoint_loading)
                .sub(CheckpointLoadedEvent, self.handle_checkpoint_loaded)
        )

    def _write_progress_bar(self,
                            epoch_index: int,
                            epochs_limit: int,
                            loader_name: str,
                            current_text: str,
                            bar_length: int = 20):
        division = float(epoch_index) / epochs_limit
        percent = int(100 * division)
        arrow = '=' * int(round(division * bar_length) - 1) + '>'
        spaces = '_' * (bar_length - len(arrow))

        if 0 <= percent < 50:
            percent_color_f, percent_color_b = Color.F_LightYellow, Color.B_LightYellow
        elif 50 <= percent < 75:
            percent_color_f, percent_color_b = Color.F_LightRed, Color.B_LightRed
        else:
            percent_color_f, percent_color_b = Color.F_Red, Color.B_Red

        epoch_str = f"\repoch {percent_color_f}{epoch_index:0>3}{Color.F_Default}/{epochs_limit:0>3} "
        if loader_name == LoaderName.Train:
            loader_str = f"({Color.F_Cyan}{loader_name}{Color.F_Default}) "
        else:
            loader_str = f"({Color.F_Magenta}{loader_name}{Color.F_Default}) "

        percent_str = f"[{percent_color_b}{arrow}{Color.B_Default}{spaces}] "

        text = f"{epoch_str}{loader_str}{percent_str} | {current_text} | {self._info}"
        self._write(text)

    def _write(self, text: str):
        if self._mode == ConsoleMode.SingleLine:
            sys.stdout.write(text)
        elif self._mode == ConsoleMode.Basic:
            sys.stdout.write(f'{text}\n')
        else:
            raise ValueError(f'unknown mode={self._mode}')

        sys.stdout.flush()

    def _write_progress(self):
        self._write_progress_bar(epoch_index=self._current_epoch_index,
                                 epochs_limit=self._current_epochs_limit,
                                 loader_name=self._current_loader_name,
                                 current_text=self._current_text)

    def handle_runner_start(self, event: RunnerStartEvent):
        self._current_epochs_limit = event.runner.epochs_limit

    def handle_runner_force_stop(self, event: RunnerForceStopEvent):
        text = f'runner.force_stop.reason={event.reason}'
        print(text)

    def handle_epoch_start(self, event: EpochStartEvent):
        self._current_epoch_index = event.epoch.epoch_index

    def handle_loader_start(self, event: LoaderStartEvent):
        self._current_loader_name = event.loader.name

        self._write_progress()

    def handle_loader_process_batch_end(self, event: LoaderProcessBatchEndEvent):
        if event.loader.name not in self._current_global_steps:
            self._current_global_steps[event.loader.name] = 0

        self._current_global_steps[event.loader.name] += 1

    def handle_metric_manager_flush(self, event: MetricManagerFlushEvent):
        text = ""

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

        self._current_text = text

    def handle_checkpoint_saving(self, event: CheckpointSavingEvent):
        self._info = f'checkpoint_saving.{event.checkpoint_type}'

        self._write_progress()

    def handle_checkpoint_saved(self, event: CheckpointSavedEvent):
        self._info = f'checkpoint_saved.{event.checkpoint_type}'

        self._write_progress()

    def handle_checkpoint_loading(self, event: CheckpointLoadingEvent):
        self._info = f'checkpoint_loading.{event.checkpoint_type}'

        self._write_progress()

    def handle_checkpoint_loaded(self, event: CheckpointLoadedEvent):
        self._info = f'checkpoint_loaded.{event.checkpoint_type}'

        self._write_progress()
