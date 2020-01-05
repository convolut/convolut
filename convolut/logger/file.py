import logging
import os
import datetime
from .console import ConsoleLogger
from ..constants import ConsoleMode
from ..settings import LOGGER_FILE_FOLDER, LOGGER_FILE_FILENAME


class FileLogger(ConsoleLogger):
    def __init__(self,
                 folder: str = LOGGER_FILE_FOLDER,
                 filename: str = LOGGER_FILE_FILENAME,
                 ):
        super().__init__(mode=ConsoleMode.Basic)

        self._folder = folder
        os.makedirs(self._folder, exist_ok=True)

        self._filename = filename
        self._logfile = os.path.join(self._folder, self._filename)

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self._logfile)
        file_handler.setLevel(logging.INFO)

        self._logger.addHandler(file_handler)

    def _write_progress_bar(self,
                            epoch_index: int,
                            epochs_limit: int,
                            loader_name: str,
                            current_text: str,
                            bar_length: int = 20):
        epoch_str = f"epoch {epoch_index:0>3}/{epochs_limit:0>3} "
        loader_str = f"({loader_name}) "

        text = f"{datetime.datetime.utcnow()} {epoch_str}{loader_str} {current_text}"
        self._write(text)

    def _write(self, text: str):
        self._logger.info(text)
