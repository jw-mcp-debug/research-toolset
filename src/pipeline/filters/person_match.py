"""
Person hallucination filter (deterministic).

Discards extracts from sources that do not mention the target person.
The decision whether a request is about a person at all is not made
here: it comes from the query-anchor classifier
(`src/pipeline/classifiers/query_anchor.py`); this filter only runs when
the classifier has recognised a person with confidence >= 0.7.

Contradictory or merely "does not cover the topic" extracts are not
handled by a phrase list either: the harvest prompt marks every extract
as POSITIVE / NEGATIVE / META (`polarity` field), and filtering uses
that field.

This module therefore contains ONLY the deterministic token-match
function. All meaning-based decisions are left to the classifiers.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from src.pipeline.filters.base import FilterStats

logger = logging.getLogger(__name__)


# ─── Deterministic token-match functions ───────────────────────


def normalize_for_match(text: str) -> str:
    """Lower-cased, accent-free, whitespace-normalised.

    This normalisation is necessary because sources may spell person
    names in different ways:
      - 'Müller' vs 'Mueller' (German umlaut convention outside Germany)
      - 'Schöne grüße' vs 'schoene gruesse'
      - 'Straße' vs 'Strasse'
      - 'Schmidt-Jensen' vs 'Schmidt Jensen'

    Strategy: first convert German umlauts AND ß explicitly, then NFKD
    for the remaining diacritics (French, Polish, etc.), then lower-case.
    """
    # Order matters: umlauts BEFORE NFKD, otherwise they are decomposed into
    # letter + diaeresis and we get "u" instead of "ue".
    # Lower-case first so that ÜÄÖ are covered too.
    text = text.lower()
    replacements = {
        "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Remove the remaining diacritics (e.g. é, ñ, č)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\s+", " ", text)
    return text


# Honorifics and titles — ignored in the token comparison because they
# do not help to identify a person and sources often omit them or
# spell them differently (Prof. vs Professor, Dr. vs Doktor).
HONORIFICS = frozenset({
    "dr", "drs", "prof", "professor", "professorin",
    "doktor", "doktorin", "doc", "docent",
    "mr", "mrs", "ms", "miss", "herr", "frau",
    "sir", "lord", "lady", "phd", "md",
})


def _tokens(text: str) -> set[str]:
    """Tokenise a normalised text into a set of words (length > 1).

    Honorifics and titles are filtered out — they do not help to
    identify a person and would otherwise cause false negatives
    ('Dr. Schmidt' in the target vs 'Prof. Schmidt' in the content).
    """
    raw = {t for t in re.split(r"[^a-z0-9]+", text) if len(t) > 1}
    return raw - HONORIFICS


def is_person_in_source(
    target: str,
    source_content: str,
    *,
    min_token_match: int = 1,
) -> bool:
    """Heuristic token match: does the person's name occur in the source?

    This function is DETERMINISTIC — it does NOT try to guess whether
    the request is about a person. It only checks whether the name tokens
    supplied by the query-anchor classifier occur in the source.

    Strategy:
      - full normalised name as a substring → match
      - OR: all tokens of the name (except one-letter words) in the
        normalised source text → match

    Args:
        target: person target supplied by the classifier,
            e.g. "Jonas Brenner" or "Prof. Dr. Schmidt"
        source_content: full source text (NOT truncated)
        min_token_match: for multi-part names, the minimum number of
            tokens that must match. Default 1 = every token must match
            (the threshold is interpreted as "all tokens" below).

    Returns:
        True if the person is plausibly mentioned in the source.
    """
    target = (target or "").strip()
    if not target:
        # Safe default: with an empty target the filter is NOT activated
        # (the caller implemented applies_to incorrectly).
        return True

    norm_source = normalize_for_match(source_content)
    norm_target = normalize_for_match(target)

    # Strategy 1: full name as a substring
    if norm_target in norm_source:
        return True

    # Strategy 2: all tokens in the source text
    target_tokens = _tokens(norm_target)
    if not target_tokens:
        return True
    source_tokens = _tokens(norm_source)

    # We require all tokens of the name to occur — not necessarily
    # next to each other. That catches e.g. "Jonas ... Brenner" across
    # several sentences.
    matched = target_tokens & source_tokens
    if len(matched) >= max(min_token_match, len(target_tokens)):
        return True

    return False


# ─── Filter class ──────────────────────


class PersonHallucinationFilter:
    """Discard extracts from sources that do not mention the target person.

    Activation:
        Only if the query-anchor classifier returned confidence >= 0.7
        and type="person". Concretely, `applies_to` checks whether
        `ctx.query_anchor` is a person with sufficient confidence.

    Effect:
        For each extract, the source (full source content) is checked
        for a token match with the person's name. Sources without a
        match lose all their extracts.

    Important:
        The filter makes NO meaning-based decision. The decision "is
        this request about a person" has already been made by the
        classifier.
    """

    name = "person_hallucination"

    def __init__(self, source_content_lookup):
        """
        Args:
            source_content_lookup: Callable[[str], str] — returns the full
                source text for a source URL (or an empty string).
                Provided by the caller (orchestrator), because this module
                does not know where the content comes from (cache, source
                list, ...).
        """
        self.source_content_lookup = source_content_lookup

    def applies_to(self, ctx: object) -> tuple[bool, str]:
        """Active if ctx.query_anchor holds a confirmed person."""
        anchor = getattr(ctx, "query_anchor", None)
        if anchor is None:
            return False, "no query_anchor in the context"
        # Duck typing: we accept QueryAnchor objects or dicts
        if hasattr(anchor, "is_person"):
            if anchor.is_person():
                return True, (
                    f"query anchor is a person: {anchor.target!r} "
                    f"(confidence {anchor.confidence:.2f})"
                )
            return False, (
                f"query anchor is not a person (type={anchor.type}, "
                f"confidence={anchor.confidence:.2f})"
            )
        # dict variant
        if (anchor.get("type") == "person"
                and anchor.get("confidence", 0.0) >= 0.7
                and (anchor.get("target") or "").strip()):
            return True, f"query anchor is a person: {anchor.get('target')}"
        return False, "query anchor is not a person"

    async def apply(
        self, extracts: list, ctx: object,
    ) -> tuple[list, list, FilterStats]:
        """Apply the token match to the extracts."""
        anchor = getattr(ctx, "query_anchor", None)
        target = (
            getattr(anchor, "target", None)
            if anchor is not None and not isinstance(anchor, dict)
            else (anchor or {}).get("target", "")
        )

        stats = FilterStats(
            name=self.name,
            activated=True,
            activation_reason=f"target={target!r}",
            total=len(extracts),
        )

        if not target:
            # Safety net: no target → filter nothing
            stats.kept = len(extracts)
            return list(extracts), [], stats

        # Per-source cache, because several extracts may come from the same source
        source_match_cache: dict[str, bool] = {}

        kept = []
        rejected = []
        for ex in extracts:
            source_url = (
                ex.get("source_url", "") if isinstance(ex, dict)
                else getattr(ex, "source_url", "")
            )
            if source_url not in source_match_cache:
                content = self.source_content_lookup(source_url) or ""
                source_match_cache[source_url] = is_person_in_source(
                    target, content,
                )
            if source_match_cache[source_url]:
                kept.append(ex)
            else:
                rejected.append(ex)
                stats.rejection_reasons.append(
                    f"{source_url}: no token match for {target!r}"
                )

        stats.kept = len(kept)
        stats.rejected = len(rejected)
        return kept, rejected, stats
