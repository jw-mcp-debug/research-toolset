"""
StopSignal — shared stop token for all pipelines.

A single value object instead of a boolean attribute that is handed to
sub-pipelines via lambdas: with a bool, every pipeline has to hard-code
its own stop logic, and new pipelines get forgotten.

StopSignal:
  - is passed by reference through all pipelines (same instance)
  - is set by the UI's stop button
  - is checked by the pipeline at key points
  - is thread-safe via a lock (in practice asyncio is enough)

Usage:
    from src.core.stop_signal import StopSignal

    signal = StopSignal()              # one per run
    # stop button:
    signal.request_stop("user_clicked_stop")
    # in the pipeline:
    if signal.is_stop_requested():
        signal.raise_if_stopped()      # raises asyncio.CancelledError
"""

import asyncio
import threading
from dataclasses import dataclass, field


@dataclass
class StopSignal:
    """Shared stop token for the research pipeline.

    One instance per run. A run comprises all sub-pipelines that belong
    to the same user click (format → plan → search → fetch → harvest → ...).

    Why a class instead of a plain bool: we want to store a reason (for
    logging/UI), and the class is thread-safe via a lock — even though in
    practice we only need asyncio.
    """

    _stopped: bool = field(default=False, init=False)
    _reason: str = field(default="", init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def request_stop(self, reason: str = "user_request") -> None:
        """Mark the pipeline as stopped.

        Idempotent: repeated calls are fine, the first reason wins
        (typically "user_request" from the UI).
        """
        with self._lock:
            if not self._stopped:
                self._stopped = True
                self._reason = reason

    def is_stop_requested(self) -> bool:
        """True if a stop has been requested."""
        with self._lock:
            return self._stopped

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def raise_if_stopped(self) -> None:
        """Raise asyncio.CancelledError if a stop has been requested.

        Call at key points in the pipeline (e.g. between phases). asyncio
        handles the exception and cleanly aborts the running coroutine.
        """
        if self.is_stop_requested():
            raise asyncio.CancelledError(
                f"Research stopped: {self.reason}"
            )

    def reset(self) -> None:
        """Reset the signal (for reuse).

        In practice one rather creates a new instance per run; reset()
        exists for UI cases where the same StopSignal state object is
        reused between runs.
        """
        with self._lock:
            self._stopped = False
            self._reason = ""

    def __bool__(self) -> bool:
        """So that `if signal:` means "stop requested"."""
        return self.is_stop_requested()
