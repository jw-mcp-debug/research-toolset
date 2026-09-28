"""
Robust JSON parser for LLM answers.

LLMs deliver JSON in many variants:
  - bare: `{"key": "value"}`
  - in a Markdown code block: ```json
{...}
```
  - with prefix text: "Here is the answer: {...}"
  - preceded by <think> tags (Qwen, DeepSeek)
  - cut off at the end by max_tokens

The classifier calls need this parsing in many places, hence a module of
its own.

Main function:
    parse_llm_json(text, expected_keys=None, default=None) -> dict | list

Strategies, tried in this order:
  1. directly: `json.loads(text)`
  2. Markdown block: extract from ```json ... ```
  3. curly braces: from the first `{` to the matching `}`
     (nested too, with a bracket counter)
  4. square brackets: first `[` to the last `]` (for JSON arrays)
  5. repair: JSON truncated by a max_tokens cutoff
"""

import json
import logging
import re
from typing import Any, Iterable, Optional, Union

logger = logging.getLogger(__name__)

JSONValue = Union[dict, list, str, int, float, bool, None]


# ─── Strategies ────────────────────────────────────────────────────


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and unfinished think openings.

    Qwen/DeepSeek models often emit reasoning in <think> tags first and
    the JSON afterwards. With a max_tokens cutoff the closing tag is
    sometimes missing — hence `<think>.*$` is stripped as well.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def _try_direct(text: str) -> Optional[JSONValue]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _try_markdown_block(text: str) -> Optional[JSONValue]:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _try_braced_object(text: str) -> Optional[dict]:
    """First `{` to the matching `}` (bracket counter, respects strings)."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    # Brackets not closed — no match
    return None


def _try_bracketed_array(text: str) -> Optional[list]:
    """First `[` to the matching `]` (same logic as for objects)."""
    start = text.find("[")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _try_repair_truncated(text: str) -> Optional[dict]:
    """Repair a truncated top-level object.

    Common pattern: max_tokens cut in the middle of a string. We try to
    cut backwards from the end until we hit a completely parsable prefix.
    """
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    text = text[start:]

    # Try appending suffixes that close the object
    for suffix in ["}", '"}', '"]}', '"}]}', "]}}", '"" }}']:
        try:
            result = json.loads(text + suffix)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            continue

    # More aggressive: cut at the last comma, then try the suffixes
    for cut in range(1, min(500, len(text))):
        candidate = text[:-cut].rstrip().rstrip(",")
        if not candidate.endswith("}") and not candidate.endswith("]"):
            for suffix in ["}", '"}', '"]}', '"}]}']:
                try:
                    result = json.loads(candidate + suffix)
                    if isinstance(result, dict) and result:
                        return result
                except (json.JSONDecodeError, ValueError):
                    continue

    return None


# ─── Public API ────────────────────────────────────────────────────


def parse_llm_json(
    text: str,
    expected_keys: Optional[Iterable[str]] = None,
    default: Any = None,
) -> Any:
    """Parse JSON from an LLM answer, robust against various formats.

    Strategy order:
        1. directly
        2. from a Markdown block
        3. from the first {…} (with a bracket counter)
        4. from the first […] (for JSON arrays)
        5. repair of truncated objects
        — everything twice: once with the original text, once after
        stripping <think> tags

    Args:
        text: raw LLM output.
        expected_keys: if set, the result MUST be a dict and contain at
            least these keys. Otherwise `default` is returned.
        default: what to return if nothing can be parsed. Default is
            `{}` when an object is expected, `None` otherwise.

    Returns:
        The parsed structure or `default`.

    Convention for classifiers: on failure the caller should use the
    classifier's conservative default. That means callers check
    explicitly `if not result: ...`.
    """
    if default is None:
        default = {} if expected_keys else None

    if not text or not text.strip():
        return default

    candidates = [text]
    stripped = _strip_think_tags(text)
    if stripped != text and stripped:
        candidates.append(stripped)

    for raw in candidates:
        # Strategy order: with bare JSON, the character that comes first
        # decides whether we try an object or an array. That catches cases
        # like "Here are the items: [...]" where an inner `{` in an array
        # element would otherwise wrongly match before the `[`.
        first_brace = raw.find("{")
        first_bracket = raw.find("[")
        if first_bracket >= 0 and (first_brace < 0 or first_bracket < first_brace):
            structural = (_try_bracketed_array, _try_braced_object)
        else:
            structural = (_try_braced_object, _try_bracketed_array)

        for strategy in (
            _try_direct,
            _try_markdown_block,
            *structural,
            _try_repair_truncated,
        ):
            result = strategy(raw)
            if result is None:
                continue
            if expected_keys is not None:
                if not isinstance(result, dict):
                    continue
                if not all(k in result for k in expected_keys):
                    # Schema violation: try the next strategy
                    logger.debug(
                        "JSON parsed, but expected_keys are missing: "
                        f"{set(expected_keys) - set(result.keys())}"
                    )
                    continue
            return result

    logger.warning(
        "JSON parsing failed "
        f"(length={len(text)}, "
        f"has_braces={'{' in text}, "
        f"has_brackets={'[' in text}, "
        f"has_think={'<think>' in text}): "
        f"{text[:300]!r}"
    )
    return default


def coerce_float(value: Any, default: float = 0.0,
                 lo: float = 0.0, hi: float = 1.0) -> float:
    """Robust conversion of an LLM value to float[lo,hi].

    LLMs sometimes deliver confidence values as a string ("0.85"), as a
    percentage ("85%") or as an int (1). We normalise to [0,1].
    """
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        s = value.strip().rstrip("%").replace(",", ".")
        try:
            f = float(s)
            if "%" in value or f > hi * 1.5:  # 85 → 0.85
                f /= 100.0
        except ValueError:
            return default
    else:
        return default

    if f < lo:
        return lo
    if f > hi:
        return hi
    return f
