"""Optional enhancement lifecycle events, independent of command presentation."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

ProgressSink = Callable[[str, Literal["started", "finished", "failed"]], None]


@contextmanager
def progress_stage(sink: ProgressSink | None, stage: str) -> Iterator[None]:
    """Bracket actual work; completion is not a processing or quality verdict."""
    if sink is not None:
        sink(stage, "started")
    try:
        yield
    except BaseException:
        if sink is not None:
            sink(stage, "failed")
        raise
    else:
        if sink is not None:
            sink(stage, "finished")
