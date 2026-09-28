"""
Content-based de-duplication of sources.

The orchestrator uses it after URL-based de-duplication, to recognise
mirror subdomains and mirrored CMS trees that have different URLs but
deliver identical content. Typical example: an institution's CMS serving
the same pages under several host names (committees.example.edu,
boards.example.edu, representatives.example.edu).

Strategy: compute a compact "content fingerprint" for every source (hash
over the normalised content) and collapse sources with the same
fingerprint — the one with the "most canonical" URL remains.

Heuristic for "canonical": the shorter URL wins, with equal length the
alphabetically first — which prefers main domains over subdomain
mirrors (`example.edu` before `cms.example.edu`).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)


# Minimum content length: shorter content is NOT compared
# (too little substance to count reliably as a duplicate).
_MIN_CONTENT_FOR_FINGERPRINT = 200

# How many characters from the start go into the fingerprint. More than
# 4000 characters rarely adds distinction; fewer than 1000 misses real
# differences in CMS trees that show the same sidebar at the bottom
# but different main content at the top.
_FINGERPRINT_PREFIX_LENGTH = 4000


def _normalize_for_fingerprint(content: str) -> str:
    """Normalise content for a robust fingerprint comparison.

    - lower case
    - collapse whitespace runs to a single space
    - remove URLs (they differ on mirror domains)
    - trim to the leading segment
    """
    if not content:
        return ""
    s = content.lower()
    # Remove URLs — they contain the host name and differ between
    # mirror subdomains without being semantically relevant.
    s = re.sub(r"https?://\S+", "", s)
    # Normalise whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s[:_FINGERPRINT_PREFIX_LENGTH]


def _content_fingerprint(content: str) -> str:
    """Return a 16-character hex fingerprint of the content.

    Identical hashes → identical normalised content.
    With too little content (< _MIN_CONTENT_FOR_FINGERPRINT) an empty
    string is returned — such sources are not de-duplicated.
    """
    normalized = _normalize_for_fingerprint(content)
    if len(normalized) < _MIN_CONTENT_FOR_FINGERPRINT:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _canonicality_key(url: str) -> tuple:
    """Sort key for picking the "most canonical" URL.

    Lower value = better. Order:
      1. URL length (shorter → better, prefers main domains)
      2. number of dots in the host name (fewer subdomains → better)
      3. alphabetical (deterministic on ties)
    """
    if not url:
        return (10**9, 10**9, "")
    # Extract the host name (rough — avoids urllib.parse imports)
    # Example: 'https://cms.example.edu/path' → 'cms.example.edu'
    m = re.match(r"https?://([^/]+)", url)
    host = m.group(1) if m else url
    return (len(url), host.count("."), url.lower())


def dedupe_sources_by_content(sources: list) -> Tuple[list, int]:
    """Remove sources with identical content.

    Args:
        sources: list of SourceDocument objects (duck-typed: only needs
                 .url and .content).

    Returns:
        (kept_sources, removed_count):
          - kept_sources: list of the surviving sources in their original
            order. Per fingerprint group the most canonical URL is kept;
            all others are removed.
          - removed_count: number of removed mirror duplicates.

    Sources with too little content (< _MIN_CONTENT_FOR_FINGERPRINT) are
    NOT de-duplicated — they survive unchanged. This prevents several
    short stub pages from being collapsed by mistake.
    """
    if not sources:
        return [], 0

    # First pass: compute fingerprints and form groups
    fingerprint_groups: dict[str, list] = {}
    no_fingerprint: list = []  # sources without a fingerprint (too short)

    for src in sources:
        content = getattr(src, "content", "") or ""
        fp = _content_fingerprint(content)
        if not fp:
            no_fingerprint.append(src)
        else:
            fingerprint_groups.setdefault(fp, []).append(src)

    # Second pass: pick the most canonical source per group
    survivors: set = set()  # id() of the surviving sources
    for fp, group in fingerprint_groups.items():
        if len(group) == 1:
            survivors.add(id(group[0]))
            continue
        # Several mirror sources — pick the most canonical
        canonical = min(group, key=lambda s: _canonicality_key(s.url))
        survivors.add(id(canonical))
        # Log the removed mirrors
        for src in group:
            if id(src) != id(canonical):
                logger.debug(
                    "Content dedup: %s is a mirror of %s (same content)",
                    src.url, canonical.url,
                )

    # Third pass: reassemble in the original order
    kept = []
    for src in sources:
        if src in no_fingerprint:
            kept.append(src)
        elif id(src) in survivors:
            kept.append(src)

    removed = len(sources) - len(kept)
    return kept, removed
