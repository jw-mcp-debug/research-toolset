"""
Harvest response parser with polarity markers.

The harvest prompt asks the LLM to mark every extract explicitly as
POSITIVE / NEGATIVE / META. This parser reads the markers and sets the
`polarity` field of the SourceExtract. Markers and field labels are
accepted in English and German, because the model may answer in the
report language.

Examples of the format that is parsed:

    [F1] [POSITIVE] Fact: ...
         Context: ...
         Reliability: high

    [F2] [NEGATIVE] Fact: the source does not cover the topic in detail
         Context: ...

    [F3] [META] Fact: the source is marketing material, not a datasheet
         Context: ...

If the marker is missing, the extract is treated as POSITIVE — a
conservative default, nothing is lost.
"""

from __future__ import annotations

import logging
from src.core.vocab import MEDIUM, normalize_level
import re

from src.pipeline.models import SourceExtract

logger = logging.getLogger(__name__)


# Pattern for the question-ID line with an optional polarity marker.
# Accepts (tried in this order):
#   [F1] [POSITIVE] Fact: text
#   [F1] Fact: text
#   F1 [POSITIVE]: text
#   F1: text
_QID_LINE_RE = re.compile(
    r"^\[?(F\d+)\]?\s*"            # F1 or [F1]
    r"(?:\[(POSITIVE?|NEGATIVE?|META)\]\s*)?"   # optional polarity
    r"(?:Fakt|Fact)?\s*:?\s*"
    r"(.*)$",
    re.IGNORECASE,
)

_CONTEXT_RE = re.compile(
    r"^(?:Kontext|Context)\s*:?\s*(.*)$", re.IGNORECASE,
)

_RELIABILITY_RE = re.compile(
    r"^(?:Verl(?:ä|ae)sslichkeit|Reliability|Verl\.?)"
    r"\s*:?\s*(hoch|mittel|niedrig|high|medium|low)",
    re.IGNORECASE,
)

# Kept for callers that look values up directly; normalisation lives in
# src.core.vocab.
_RELIABILITY_MAP = {
    "hoch": "high", "mittel": "medium", "niedrig": "low",
}

_POLARITY_MAP = {
    "POSITIV": "positive", "POSITIVE": "positive",
    "NEGATIV": "negative", "NEGATIVE": "negative",
    "META": "meta",
}


def parse_harvest_response_with_polarity(
    response: str,
    source_url: str,
    source_title: str,
) -> list[SourceExtract]:
    """Parse the harvest answer including polarity markers.

    Args:
        response: raw LLM answer of the harvest call.
        source_url, source_title: source metadata for the extracts.

    Returns:
        List of SourceExtract objects with the `polarity` field set.
    """
    extracts: list[SourceExtract] = []

    current_qid: str = ""
    current_polarity: str = "positive"
    current_fact: str = ""
    current_context: str = ""
    current_reliability: str = MEDIUM

    def flush() -> None:
        if current_qid and current_fact:
            extracts.append(SourceExtract(
                source_url=source_url,
                source_title=source_title,
                question_id=current_qid,
                fact=current_fact.strip(),
                context=current_context.strip(),
                reliability=current_reliability,
                polarity=current_polarity,
            ))

    for raw_line in response.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Remove the bullet prefix
        line = re.sub(r"^[-*•]\s*", "", line)

        # New question entry?
        qid_match = _QID_LINE_RE.match(line)
        if qid_match and qid_match.group(1):
            # Commit the previous entry
            flush()
            current_qid = qid_match.group(1).upper()
            polarity_marker = qid_match.group(2)
            if polarity_marker:
                current_polarity = _POLARITY_MAP.get(
                    polarity_marker.upper(), "positive",
                )
            else:
                # No marker → default POSITIVE
                current_polarity = "positive"
            current_fact = qid_match.group(3).strip()
            current_context = ""
            current_reliability = MEDIUM
            continue

        # Context line
        ctx_match = _CONTEXT_RE.match(line)
        if ctx_match:
            current_context = ctx_match.group(1).strip()
            continue

        # Reliability line
        rel_match = _RELIABILITY_RE.match(line)
        if rel_match:
            val = rel_match.group(1).lower()
            current_reliability = normalize_level(val)
            continue

        # Otherwise: continuation of the current fact
        if current_qid and not line.startswith("["):
            current_fact = (current_fact + " " + line).strip()

    flush()
    return extracts


def count_by_polarity(extracts: list[SourceExtract]) -> dict[str, int]:
    """Count extracts by polarity (for logging/UI/diagnosis).

    Returns:
        {"positive": int, "negative": int, "meta": int}
    """
    counts = {"positive": 0, "negative": 0, "meta": 0}
    for e in extracts:
        pol = getattr(e, "polarity", "positive")
        if pol in counts:
            counts[pol] += 1
        else:
            counts["positive"] += 1  # unknown → positive (Default)
    return counts


def positive_only(extracts: list[SourceExtract]) -> list[SourceExtract]:
    """Helper: only the positive extracts, for the synthesis."""
    return [e for e in extracts if getattr(e, "polarity", "positive") == "positive"]


def meta_only(extracts: list[SourceExtract]) -> list[SourceExtract]:
    """Helper: only the META extracts, for the methodology section."""
    return [e for e in extracts if getattr(e, "polarity", "positive") == "meta"]
