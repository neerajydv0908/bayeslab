"""Optional wall-clock limits for long exact calculations.

Exact inference is exponential in the worst case, so a server that accepts
models from strangers needs a way to stop a calculation that runs too long.
The heavy loops call check_time(), which costs one ContextVar lookup when no
limit is active. Limits apply to the current thread (or asyncio task) only.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import math
import time

from .model import ModelError

_deadline = ContextVar("bayeslab_deadline", default=None)


class ComputationTimeout(ModelError):
    """A calculation ran past its time limit and was stopped."""


@contextmanager
def time_limit(seconds):
    """Stop calculations inside this block after `seconds` (None means no limit).

    A nested limit can only shorten the time left, never extend it.
    """
    if seconds is None:
        yield
        return
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("A time limit must be a positive number of seconds.")
    deadline = time.monotonic() + seconds
    current = _deadline.get()
    if current is not None and current[0] <= deadline:
        deadline, seconds = current
    token = _deadline.set((deadline, seconds))
    try:
        yield
    finally:
        _deadline.reset(token)


def check_time():
    """Raise ComputationTimeout if the active time limit has passed."""
    limit = _deadline.get()
    if limit is not None and time.monotonic() > limit[0]:
        raise ComputationTimeout(
            f"This calculation was stopped after the {limit[1]:g}-second time limit. Try adding evidence or "
            "simplifying the network; with enumeration, variable elimination is usually much faster.")
