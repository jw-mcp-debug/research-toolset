"""
TaskRateLimiter: rate limiter for LLM calls in the analysis pipeline.

Differs from the connector rate limiter (for external APIs):
- token-bucket algorithm with a configurable rate
- exponential back-off on server errors (doubling up to a cap)
- async-capable, with asyncio.Lock for safety
- back-off is reset after successful calls
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class TaskRateLimiter:
    """Token-bucket rate limiter with back-off.

    Default configuration: 3 calls/s with a burst capacity of 5,
    back-off starting at 1 s, at most 60 s.

    Usage:
        limiter = TaskRateLimiter(rate=3.0)
        async def make_call():
            await limiter.acquire()
            try:
                result = await llm.complete(...)
                # success: reset the back-off
                limiter.report_success()
                return result
            except RateLimitError:
                await limiter.report_rate_limited()
                raise  # the caller decides about retrying
    """

    def __init__(
        self,
        rate: float = 3.0,                # calls per second
        burst: int = 5,                   # maximum tokens in the bucket
        backoff_initial: float = 1.0,
        backoff_max: float = 60.0,
        backoff_factor: float = 2.0,
    ):
        self.rate = rate
        self.burst = burst
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.backoff_factor = backoff_factor

        self._tokens: float = float(burst)
        self._last_refill: float = time.monotonic()
        self._current_backoff: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            await self._wait_for_token()

    async def _wait_for_token(self) -> None:
        """Wait (inside the lock)."""
        # If a back-off is currently running: wait
        if self._current_backoff > 0:
            wait = self._current_backoff
            logger.debug(f"Rate limiter back-off: waiting {wait:.1f}s")
            await asyncio.sleep(wait)

        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # waiting time until the next token
            needed = 1.0 - self._tokens
            wait = needed / self.rate
            await asyncio.sleep(wait)

    def _refill(self) -> None:
        """Refill the token bucket based on the elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            float(self.burst),
            self._tokens + elapsed * self.rate,
        )
        self._last_refill = now

    def report_success(self) -> None:
        """Reset the back-off after a successful call."""
        if self._current_backoff > 0:
            logger.debug("Rate limiter: back-off reset")
        self._current_backoff = 0.0

    async def report_rate_limited(self) -> None:
        """Increase the back-off (call on 429/server errors).

        Doubles the waiting time, starting at backoff_initial.
        Capped at backoff_max.
        """
        if self._current_backoff == 0:
            self._current_backoff = self.backoff_initial
        else:
            self._current_backoff = min(
                self.backoff_max,
                self._current_backoff * self.backoff_factor,
            )
        logger.warning(
            f"Rate limiter: back-off raised to {self._current_backoff:.1f}s"
        )

    @property
    def current_backoff_seconds(self) -> float:
        return self._current_backoff
