"""
Adapter between `DualLLMClient` and the `ClassifierLLM` protocol.

The `DualLLMClient` (src/llm/client.py) has two models, primary and
harvest. Classifiers are lightweight calls and use the harvest LLM by
default (faster, cheaper).

This class wraps `DualLLMClient` so that it satisfies the
`ClassifierLLM` protocol — `complete(messages, max_tokens) -> str`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.client import DualLLMClient


class HarvestModelAdapter:
    """Wraps `DualLLMClient.harvest_complete` as `ClassifierLLM.complete`."""

    def __init__(self, dual_llm: "DualLLMClient"):
        self._dual = dual_llm

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> str:
        return await self._dual.harvest_complete(messages, max_tokens=max_tokens)
