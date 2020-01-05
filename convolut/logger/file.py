import logging
import os

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
        self._filename = filename
        self._logfile = os.path.join(self._folder, self._filename)

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self._logfile)
        file_handler.setLevel(logging.INFO)

        self._logger.addHandler(file_handler)

    def _write(self, text: str):
        self._logger.info(text)
