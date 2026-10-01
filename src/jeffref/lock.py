"""One model in memory at a time: every command that loads a model (bench-27b, bench-jeff) holds an exclusive lock on
models/.one-model.lock while it runs, and a second such command fails at once instead of loading beside it."""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager

from jeffref.tasks import MODELS

LOCK = MODELS / ".one-model.lock"


@contextmanager
def one_model(what: str) -> Iterator[None]:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            raise RuntimeError(f"Another benchmark command holds a model in memory ({handle.read().strip()}); "
                               f"wait for it to finish before starting {what}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(what + "\n")
        handle.flush()
        yield
