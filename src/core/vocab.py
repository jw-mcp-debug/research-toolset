"""Fixed codes for values that the LLM returns and the code acts on.

Priorities, reliabilities and similar ratings drive limits, ordering and
colours. They are stored as fixed English codes. The LLM may answer in
the report language, so every value read from a model response goes
through the normalisers below, which accept German and English words.
"""

from __future__ import annotations

HIGH, MEDIUM, LOW = "high", "medium", "low"
LEVELS = (HIGH, MEDIUM, LOW)

_LEVEL_ALIASES = {
    "high": HIGH, "hoch": HIGH, "h": HIGH,
    "medium": MEDIUM, "mittel": MEDIUM, "moderate": MEDIUM, "m": MEDIUM,
    "low": LOW, "niedrig": LOW, "gering": LOW, "l": LOW,
}

STRONG, MODERATE, WEAK, UNKNOWN = "strong", "moderate", "weak", "unknown"
_QUALITY_ALIASES = {
    "strong": STRONG, "stark": STRONG, "high": STRONG, "hoch": STRONG,
    "moderate": MODERATE, "medium": MODERATE, "mittel": MODERATE,
    "weak": WEAK, "schwach": WEAK, "low": WEAK, "niedrig": WEAK,
}


def normalize_level(value, default: str = MEDIUM) -> str:
    """'hoch'/'High'/'medium' … → 'high' | 'medium' | 'low'."""
    key = str(value or "").strip().strip("\"'").lower()
    return _LEVEL_ALIASES.get(key, default)


def normalize_quality(value) -> str:
    """Overall quality rating → 'strong' | 'moderate' | 'weak' | 'unknown'.

    Matches on the first known word, so free text such as
    "moderate (small sample)" is recognised.
    """
    text = str(value or "").lower()
    for word, code in _QUALITY_ALIASES.items():
        if word in text:
            return code
    return UNKNOWN
