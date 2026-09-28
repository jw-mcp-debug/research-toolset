"""
Test helpers for the research pipeline.

The central tool is `MockLLM` — an async LLM stub with a response queue.
Tests fill the queue with the answers the LLM "would give", and the code
under test receives them in turn. That makes classifier tests fast,
deterministic and independent of a real LLM endpoint.

Written in stdlib `unittest` style, so the tests also run without pytest
(pytest recognises unittest.TestCase automatically).

Every classifier is tested along four axes:
  - happy path with a clear answer
  - low-confidence answer → fallback active
  - schema violation in the LLM output → graceful fallback
  - reasoning consistency: with confidence > 0.8, reasoning must be > 50 characters

These four axes are covered in the classifier test modules
(test_classifier_*.py).
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ─── Mock-LLM ──────────────────────────────────────────────────────


@dataclass
class MockLLM:
    """LLM stub for tests.

    Usage:
        llm = MockLLM()
        llm.queue_json({"anchor_type": "none", "confidence": 0.94, ...})
        llm.queue_text("free answer")  # for non-JSON calls

        result, call = await classify_query_anchor("...", "...", llm)

    If the queue is empty at the call, `default_response` is returned. The
    default is an empty string — useful for tests in which we deliberately
    test fallback behaviour.

    `recorded_calls` records every call (messages + max_tokens), so that
    tests can check what the LLM was called with.
    """

    responses: deque = field(default_factory=deque)
    default_response: str = ""
    recorded_calls: list = field(default_factory=list)
    raise_on_call: Optional[Exception] = None
    delay_seconds: float = 0.0

    def queue(self, response: str) -> "MockLLM":
        """Put a raw text answer into the queue."""
        self.responses.append(response)
        return self

    def queue_json(self, obj) -> "MockLLM":
        """Put a JSON-serialisable dict/list into the queue.

        The mock returns it as a JSON string — the code under test uses
        its own JSON parser.
        """
        self.responses.append(json.dumps(obj, ensure_ascii=False))
        return self

    def queue_markdown_json(self, obj) -> "MockLLM":
        """Like queue_json, but in a Markdown code block — tests parser strategy 2."""
        self.responses.append(f"```json\n{json.dumps(obj, ensure_ascii=False)}\n```")
        return self

    def queue_with_think(self, obj, thinking: str = "Lass mich nachdenken...") -> "MockLLM":
        """JSON preceded by a <think> block — tests strategy 4."""
        self.responses.append(
            f"<think>{thinking}</think>\n{json.dumps(obj, ensure_ascii=False)}"
        )
        return self

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> str:
        """ClassifierLLM protocol implementation."""
        self.recorded_calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
        })
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        if self.responses:
            return self.responses.popleft()
        return self.default_response


# ─── HarvestContext stub for filter tests ──────────────────────────


@dataclass
class StubContext:
    """Minimal context for filter tests.

    Real HarvestContext instances need many fields; for filter tests
    `query_anchor` is usually enough. The filters are duck-typed, they only
    access attributes.
    """
    query_anchor: object = None
    sources_by_url: dict = field(default_factory=dict)


# ─── Helper: classifier test axes ──────────────────────


def assert_classifier_result_consistency(test_case, result):
    """Check the consistency rules.

    Every classifier result with confidence > 0.8 should have reasoning
    > 50 characters (otherwise the high confidence is not plausibly
    justified). This does not apply to fallback paths
    (`fallback_used=True`).
    """
    if getattr(result, "fallback_used", False):
        return
    confidence = getattr(result, "confidence", 0.0)
    reasoning = getattr(result, "reasoning", "")
    if confidence > 0.8:
        test_case.assertGreater(
            len(reasoning), 50,
            f"confidence {confidence:.2f} > 0.8, but the reasoning is only "
            f"{len(reasoning)} characters long: {reasoning!r}",
        )


# ─── Async test runner for unittest ────────────────────────────────


def async_test(coro):
    """Decorator: turns an async method into a test usable with unittest.

    Not needed with pytest-asyncio (that works natively); stdlib unittest
    needs the wrapper.
    """
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    wrapper.__name__ = coro.__name__
    wrapper.__doc__ = coro.__doc__
    return wrapper
