from decouple import Module


class TelegramMode:
    Basic = "basic"


class TelegramLogger(Module):
    def __init__(self, token: str, channel: str, mode: str = TelegramMode.Basic):
        super().__init__()
        self._token = token
        self._channel = channel
        self._mode = mode
