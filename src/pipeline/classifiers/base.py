"""
LLM classifiers — base infrastructure.

Every classifier in this package follows the same pattern:

  1. JSON-mode call to the LLM.
  2. Robust parsing (`parse_llm_json` with `expected_keys`).
  3. Confidence validation (`coerce_float`, default 0.0).
  4. Confidence below threshold: the classifier's conservative default.
  5. Structured logging of the call (see `ClassifierCall`).
  6. Return as a typed result object (dataclass).

The concrete classifiers live in the sibling modules:
  - query_anchor.py
  - coverage.py
  - diagnose.py
  - source_relevance.py
  - factoids.py

Why a separate base module: all classifiers share logging, retry,
schema validation and the conservative-default logic. Instead of
repeating it per classifier, `ClassifierBase` provides the pattern and
`ClassifierCall` the log entry.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

from src.core.limits import (
    CLASSIFIER_CONFIDENCE_THRESHOLD,
)
from src.llm.json_parser import coerce_float, parse_llm_json

logger = logging.getLogger(__name__)


# ─── Protocol for the LLM client ──────────────────────────────────


class ClassifierLLM(Protocol):
    """Narrow interface the classifiers need.

    Deliberately minimal: only `complete(messages, max_tokens) -> str`.
    That lets tests inject a mock without building the whole
    `DualLLMClient` machinery.

    In production the class is called through an adapter that wraps
    `DualLLMClient.harvest_complete` (for light classifiers) or
    `primary_complete` (for more expensive ones) — see
    `ClassifierLLMAdapter` below.
    """

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> str:
        ...


# ─── Log entry per call ──────────────────────────────────────


@dataclass
class ClassifierCall:
    """Structured log of a classifier call.

    Collected by the orchestrator or UI and stored in the run directory
    together with the classifier outputs. In case of an error this makes
    it reproducible what the classifier got as input, what it answered
    and with which confidence.
    """
    name: str                              # e.g. "query_anchor"
    prompt_hash: str = ""                  # SHA-256 of the first 200 characters
    input_summary: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    confidence: float = 0.0
    duration_seconds: float = 0.0
    raw_response: str = ""                 # first 1000 characters, for debugging
    fallback_used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "prompt_hash": self.prompt_hash,
            "input_summary": self.input_summary,
            "output": self.output,
            "confidence": self.confidence,
            "duration_seconds": round(self.duration_seconds, 3),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


def hash_prompt(prompt: str) -> str:
    """Short hash of the prompt for reproducible logs."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ─── Base class for classifiers ──────────────────────────────


@dataclass
class ClassifierResult:
    """Marker base class for classifier outputs.

    Concrete classifiers derive from it and add typed fields. All of
    them should have a `confidence` and a `reasoning` field.
    """
    confidence: float = 0.0
    reasoning: str = ""
    fallback_used: bool = False


async def call_classifier(
    *,
    name: str,
    llm: ClassifierLLM,
    prompt: str,
    expected_keys: Iterable[str],
    max_tokens: int = 1024,
    retry_on_parse_failure: bool = True,
    input_summary: Optional[dict] = None,
) -> tuple[dict, ClassifierCall]:
    """Generic classifier call.

    Makes the LLM call, parses JSON, validates `expected_keys` and falls
    back to an empty dict on failure (the caller then applies its
    specific conservative default).

    With `retry_on_parse_failure=True` (default), a parse failure leads
    to a second attempt with a stricter "answer ONLY with JSON"
    instruction.

    Returns:
        (parsed_dict, call_log) — on failure parsed_dict is empty and
        call_log.fallback_used=True.
    """
    started = time.monotonic()
    call = ClassifierCall(
        name=name,
        prompt_hash=hash_prompt(prompt),
        input_summary=input_summary or {},
    )

    messages = [{"role": "user", "content": prompt}]
    raw = ""
    try:
        raw = await llm.complete(messages, max_tokens=max_tokens)
    except Exception as e:
        call.fallback_used = True
        call.fallback_reason = f"llm_call_failed: {type(e).__name__}: {e}"
        call.duration_seconds = time.monotonic() - started
        logger.warning("Classifier %s: LLM call failed: %s", name, e)
        return {}, call

    call.raw_response = raw[:1000]
    parsed = parse_llm_json(raw, expected_keys=list(expected_keys), default={})

    if not parsed and retry_on_parse_failure:
        # Retry with a stricter instruction (like DualLLMClient.complete_json)
        retry_msgs = messages + [
            {"role": "assistant", "content": raw[:500]},
            {"role": "user", "content": (
                "Your answer could not be parsed as JSON. "
                "Answer ONLY with a valid JSON object — without ``` code blocks, "
                "without explanations, without <think> tags. Start directly with { "
                "and end with }."
            )},
        ]
        try:
            raw2 = await llm.complete(retry_msgs, max_tokens=max_tokens)
            call.raw_response = raw2[:1000]
            parsed = parse_llm_json(
                raw2, expected_keys=list(expected_keys), default={}
            )
        except Exception as e:
            logger.warning("Classifier %s: retry failed: %s", name, e)

    call.duration_seconds = time.monotonic() - started

    if not parsed:
        call.fallback_used = True
        call.fallback_reason = "parse_failed_after_retry"
        logger.warning(
            "Classifier %s: could not parse JSON — fallback active. "
            "Raw[:200]=%r", name, raw[:200],
        )
    else:
        call.output = parsed
        call.confidence = coerce_float(parsed.get("confidence", 0.0))

    return parsed, call


# ─── Helpers for concrete classifiers ──────────────────


def confidence_below_threshold(conf: float, threshold: float | None = None) -> bool:
    """Helper for the conservative-default logic.

    Args:
        conf: measured confidence.
        threshold: threshold, default `CLASSIFIER_CONFIDENCE_THRESHOLD`.
    """
    t = threshold if threshold is not None else CLASSIFIER_CONFIDENCE_THRESHOLD
    return conf < t
