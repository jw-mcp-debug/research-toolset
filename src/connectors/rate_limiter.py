"""
Global rate limiter for access to external APIs.

Singleton — shared by all user sessions.
Each API gets its own semaphore + minimum delay.
429/503 responses are retried with exponential back-off.

Usage:
    limiter = get_rate_limiter()
    response = await limiter.request("crossref", client, "GET", url, params=params)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


# ─── API configuration ─────────────────────────────────────────

# Defaults — can be overridden via .env
_DEFAULT_LIMITS = {
    # name:        (max_parallel, min_delay_seconds)
    "crossref":    (5, 0.2),    # polite pool: 50/s allowed
    "openalex":    (8, 0.12),   # 10/s allowed
    "s2":          (1, 1.0),    # 1/s without key, 10/s with key
    "dblp":        (3, 0.33),   # fair use ~3/s
    "arxiv":       (1, 3.0),    # documented: max 1 request/3 s
    "url_check":   (3, 0.5),    # mixed, various hosts
    "web_fetch":   (5, 0.3),    # web scraper (normal research)
    "searxng":     (3, 0.2),    # local, but protects the upstream engines
}

MAX_RETRIES = 3
BASE_BACKOFF = 2.0  # seconds
JITTER_FACTOR = 0.3  # ±30 % random deviation
# Upper bound for EVERY single wait. Prevents an API with a large
# Retry-After header (e.g. S2 'Retry-After: 3600') or a high back-off
# exponent from freezing processing for minutes or hours, which makes
# a run appear to "hang".
MAX_BACKOFF = 30.0  # seconds


# ─── Rate limiter class ───────────────────────────────────────

class APIRateLimiter:
    """Global rate limiter with per-API semaphores and back-off."""

    def __init__(self):
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._min_delays: dict[str, float] = {}
        self._last_request: dict[str, float] = {}
        self._domain_last_request: dict[str, float] = {}
        self._domain_semaphore = asyncio.Semaphore(10)  # Global max

        # Load the configuration
        for name, (parallel, delay) in _DEFAULT_LIMITS.items():
            env_key = f"RATELIMIT_{name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val:
                try:
                    # Format: "requests_per_second" or "parallel,delay"
                    if "," in env_val:
                        p, d = env_val.split(",")
                        parallel, delay = int(p), float(d)
                    else:
                        rps = float(env_val)
                        parallel = max(1, int(rps))
                        delay = 1.0 / rps if rps > 0 else 1.0
                except ValueError:
                    pass

            self._semaphores[name] = asyncio.Semaphore(parallel)
            self._min_delays[name] = delay
            self._last_request[name] = 0.0

        # S2 API key — authenticated, but the limit stays at 1 req/s
        s2_key = os.environ.get("S2_API_KEY", "")
        if s2_key:
            logger.info("Rate limiter: S2 API key detected (1 req/s, authenticated)")

        logger.info(
            f"Rate limiter initialised: "
            f"{', '.join(f'{k}={v._value}' for k, v in self._semaphores.items())}"
        )

    async def request(
        self,
        api_name: str,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        max_retries: int = MAX_RETRIES,
        **kwargs,
    ) -> httpx.Response:
        """Rate-limited HTTP request with retry and back-off.

        Args:
            api_name: name of the API (e.g. "crossref", "s2", "web_fetch")
            client: httpx.AsyncClient
            method: "GET" or "POST"
            url: target URL
            max_retries: maximum retries on 429/503
            **kwargs: further httpx parameters (params, headers, etc.)

        Returns:
            httpx.Response

        Raises:
            httpx.HTTPError for non-retryable errors
        """
        sem = self._semaphores.get(api_name, self._domain_semaphore)
        min_delay = self._min_delays.get(api_name, 0.5)

        for attempt in range(max_retries + 1):
            # IMPORTANT: the semaphore is held ONLY around the actual request,
            # NOT around the back-off sleep. Otherwise a single 429 answer
            # (e.g. from S2 with semaphore=1) blocks the only slot for the full
            # waiting time and serialises ALL remaining entries behind it — a
            # main cause of apparent hangs with long lists.
            wait_before_retry: float | None = None
            reraise_exc: Exception | None = None

            async with sem:
                # Minimum delay since the last request to this API
                await self._wait_for_delay(api_name, min_delay)

                # For web requests: per-domain delay
                if api_name in ("web_fetch", "url_check"):
                    await self._wait_for_domain(url)

                try:
                    response = await client.request(method, url, **kwargs)

                    # success
                    if response.status_code < 400:
                        return response

                    # not retryable (4xx except 429)
                    if 400 <= response.status_code < 500 \
                            and response.status_code != 429:
                        return response

                    # 429 Too Many Requests
                    if response.status_code == 429:
                        retry_after = self._parse_retry_after(response)
                        wait = retry_after or (BASE_BACKOFF * (2 ** attempt))
                        wait = min(wait, MAX_BACKOFF)   # against huge Retry-After values
                        wait = self._add_jitter(wait)

                        if attempt < max_retries:
                            logger.warning(
                                f"Rate limit {api_name}: 429 → "
                                f"Retry {attempt + 1}/{max_retries} "
                                f"after {wait:.1f}s"
                            )
                            wait_before_retry = wait
                        else:
                            logger.warning(
                                f"Rate limit {api_name}: 429 → "
                                f"max retries reached"
                            )
                            return response

                    # 503 Service Unavailable
                    elif response.status_code in (503, 502):
                        wait = BASE_BACKOFF * (2 ** attempt)
                        wait = min(wait, MAX_BACKOFF)
                        wait = self._add_jitter(wait)

                        if attempt < max_retries:
                            logger.warning(
                                f"Server error {api_name}: "
                                f"{response.status_code} → "
                                f"Retry {attempt + 1}/{max_retries} "
                                f"after {wait:.1f}s"
                            )
                            wait_before_retry = wait
                        else:
                            return response

                    # other server error (≥500)
                    else:
                        return response

                except Exception as e:
                    # network error (timeout, connect, etc.)
                    _is_network_error = (
                        "Timeout" in type(e).__name__
                        or "Connect" in type(e).__name__
                        or "Network" in type(e).__name__
                    )
                    if _is_network_error and attempt < max_retries:
                        wait = BASE_BACKOFF * (2 ** attempt)
                        wait = min(wait, MAX_BACKOFF)
                        wait = self._add_jitter(wait)
                        logger.warning(
                            f"Network error {api_name}: {e} → "
                            f"Retry {attempt + 1}/{max_retries} "
                            f"after {wait:.1f}s"
                        )
                        wait_before_retry = wait
                    else:
                        reraise_exc = e

            # ── outside the semaphore: back-off/re-raise ──
            if reraise_exc is not None:
                raise reraise_exc
            if wait_before_retry is not None:
                await asyncio.sleep(wait_before_retry)
                continue

        # Should never be reached
        raise RuntimeError(f"Max retries exceeded for {url}")

    async def _wait_for_delay(self, api_name: str, min_delay: float):
        """Wait until the minimum delay since the last request has passed."""
        now = time.monotonic()
        last = self._last_request.get(api_name, 0.0)
        elapsed = now - last
        if elapsed < min_delay:
            wait = min_delay - elapsed
            wait = self._add_jitter(wait)
            await asyncio.sleep(wait)
        self._last_request[api_name] = time.monotonic()

    async def _wait_for_domain(self, url: str):
        """Per-domain delay for web requests (protection against Cloudflare blocks)."""
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            return

        min_domain_delay = float(os.environ.get(
            "RATELIMIT_WEBFETCH_PER_DOMAIN", "0.5"
        ))

        now = time.monotonic()
        last = self._domain_last_request.get(domain, 0.0)
        elapsed = now - last
        if elapsed < min_domain_delay:
            await asyncio.sleep(min_domain_delay - elapsed)
        self._domain_last_request[domain] = time.monotonic()

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        """Parse the Retry-After header (seconds or HTTP date)."""
        retry_after = response.headers.get("Retry-After", "")
        if not retry_after:
            return None
        try:
            return float(retry_after)
        except ValueError:
            pass
        # HTTP date → ignore, fall back to back-off
        return None

    @staticmethod
    def _add_jitter(delay: float) -> float:
        """Add random deviation (±30 %)."""
        jitter = delay * JITTER_FACTOR
        return delay + random.uniform(-jitter, jitter)

    def get_stats(self) -> dict:
        """Return the current semaphore utilisation."""
        return {
            name: {
                "available": sem._value,
                "max": _DEFAULT_LIMITS.get(name, (1, 1))[0],
                "min_delay": self._min_delays.get(name, 0),
            }
            for name, sem in self._semaphores.items()
        }


# ─── Singleton ─────────────────────────────────────────────────

_instance: APIRateLimiter | None = None


def get_rate_limiter() -> APIRateLimiter:
    """Return the global rate limiter (singleton).

    Created on the first call and then shared by all sessions.
    """
    global _instance
    if _instance is None:
        _instance = APIRateLimiter()
    return _instance
