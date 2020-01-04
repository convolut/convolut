from decouple import Module


class TensorboardMode:
    Basic = "basic"


class TensorboardLogger(Module):
    def __init__(self, folder: str, mode: str = TensorboardMode.Basic):
        super().__init__()
        self._folder = folder
        self._mode = mode
