"""
Reranker client (cross-encoder, `/rerank` API).

A reusable building block for the research pipeline and the person
directory.

API contract:

    POST {base_url}
    { "model": ..., "query": ..., "documents": [str, ...] }
    → { "results": [ {"index": int, "relevance_score": float}, ... ] }

Design principle — FAIL-OPEN: no error (missing API key, timeout, HTTP
error, unexpected schema) raises an exception to the outside; it leads
to `available=False` or an unchanged order instead. Reranking is a
quality *improvement*, not a hard dependency of the pipeline.
"""

from __future__ import annotations

import logging
from typing import Sequence
from src.core.tls import tls_verify

logger = logging.getLogger(__name__)


class Reranker:
    """Thin async client around a cross-encoder rerank endpoint.

    Usage:
        rr = Reranker(base_url, model, api_key)
        if rr.available:
            order = await rr.rank(query, documents)  # indices, best→worst
    """

    def __init__(self, base_url: str, model: str, api_key: str,
                 timeout: float = 30.0):
        self.base_url = (base_url or "").strip()
        self.model = model
        self.api_key = (api_key or "").strip()
        self.timeout = timeout

    @property
    def available(self) -> bool:
        # No endpoint OR no key: no reranking — fail-open.
        # A model name is required too: there is no sensible default.
        return bool(self.base_url and self.api_key and self.model)

    async def rank(
        self, query: str, documents: Sequence[str],
    ) -> list[tuple[int, float]]:
        """Rerank `documents` against `query`.

        Returns:
            List of (orig_index, score), sorted by descending score.
            When unavailable or on error: the input order with score 0.0
            (identity — the caller can safely sort by it, nothing
            changes).
        """
        n = len(documents)
        identity = [(i, 0.0) for i in range(n)]
        if not self.available or n == 0:
            return identity
        if n == 1:
            return [(0, 1.0)]

        try:
            import httpx
        except Exception:
            return identity

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, verify=tls_verify(),
            ) as client:
                resp = await client.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": list(documents),
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(
                "Reranker unreachable (%s) — order unchanged",
                type(e).__name__,
            )
            return identity

        scored: list[tuple[int, float]] = []
        try:
            for item in data.get("results", []):
                idx = int(item.get("index", -1))
                score = float(item.get("relevance_score", 0.0))
                if 0 <= idx < n:
                    scored.append((idx, score))
        except Exception as e:
            logger.warning("Reranker answer unreadable (%s)", type(e).__name__)
            return identity

        if not scored:
            return identity

        # Append indices the reranker did not return at the end
        seen = {i for i, _ in scored}
        for i in range(n):
            if i not in seen:
                scored.append((i, -1.0))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    async def order(
        self, query: str, items: list, key=lambda x: str(x),
    ) -> list:
        """Convenience: reorder `items` directly.

        `key(item)` returns the text that is ranked.
        When unavailable: `items` are returned unchanged.
        """
        if not items:
            return items
        docs = [key(it) or "" for it in items]
        ranking = await self.rank(query, docs)
        return [items[i] for i, _ in ranking]


def build_reranker_from_pipeline_config(config) -> Reranker:
    """Factory from a PipelineConfig (or any object with these fields)."""
    return Reranker(
        base_url=getattr(config, "reranker_base_url", ""),
        model=getattr(config, "reranker_model", ""),
        api_key=getattr(config, "reranker_api_key", ""),
    )
