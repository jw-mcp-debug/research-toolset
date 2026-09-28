"""
Embedder client (OpenAI-compatible, `/embeddings`).

Used for embedding-based de-duplication of extracts in the research
pipeline, instead of a word-overlap heuristic; the person directory
uses the same contract.

API contract:
    POST {base_url}/embeddings
    { "model": ..., "input": [str, ...] }
    → { "data": [ {"embedding": [float, ...]}, ... ] }

FAIL-OPEN: every error returns `None`; the caller then falls back to
the word-overlap heuristic. Embedding de-duplication is an
improvement, not a hard dependency.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence
from src.core.tls import tls_verify

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, base_url: str, model: str, api_key: str,
                 timeout: float = 30.0):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.model = model
        self.api_key = (api_key or "").strip()
        self.timeout = timeout

    @property
    def available(self) -> bool:
        # A model name is required too: there is no sensible default.
        return bool(self.base_url and self.api_key and self.model)

    async def embed(
        self, texts: Sequence[str],
    ) -> Optional[list[list[float]]]:
        """Embed a list of texts. On error or when unavailable: None
        (the caller uses its fallback)."""
        if not self.available or not texts:
            return None
        try:
            import httpx
        except Exception:
            return None
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, verify=tls_verify(),
            ) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": list(texts)},
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            out = [row["embedding"] for row in data.get("data", [])]
            if len(out) != len(texts):
                logger.warning(
                    "Embedder: %d vectors for %d texts — fallback",
                    len(out), len(texts),
                )
                return None
            return out
        except Exception as e:
            logger.warning(
                "Embedder unreachable (%s) — fallback",
                type(e).__name__,
            )
            return None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity without a numpy dependency."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def build_embedder_from_pipeline_config(config) -> Embedder:
    return Embedder(
        base_url=getattr(config, "embedder_base_url", ""),
        model=getattr(config, "embedder_model", ""),
        api_key=getattr(config, "embedder_api_key", ""),
    )
