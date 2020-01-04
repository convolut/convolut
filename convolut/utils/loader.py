from typing import Dict, List

from ..loader import Loader


def valid_every_n(train_loader: Loader, valid_loader: Loader, epochs: int, n: int) -> Dict[int, List[Loader]]:
    loaders = {}

    for i in range(1, epochs):
        loaders[i] = [train_loader]
        if i % n == 0:
            loaders[i].append(valid_loader)

    return loaders
