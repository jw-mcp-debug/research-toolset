"""
Helper functions for processing research search terms.

The orchestrator uses three functions from this module:

  - `coerce_search_terms`  — normalises the LLM output for `search_terms`
                             into two canonical forms (flat list +
                             dict per language).
  - `detect_term_language` — language heuristic for a single term.
  - `build_terms_by_lang`  — aggregates the search terms of all
                             questions by language.

Robustness principle: the analysis LLM delivers quite divergent
structures here — we accept as many as possible and only give up when
no meaningful structure can be recognised at all (rather than accepting
nothing and starting the pipeline without search terms).
"""

from __future__ import annotations

import re
from typing import Iterable, Optional


# ── Language heuristic ──────────────────────────────────────────────


# German markers: umlauts, ß, typical German function words.
# English is the default — mainly to decide whether a term belongs in
# the DE or the EN bucket when the LLM did not supply a language
# code.
_DE_DIACRITICS = re.compile(r"[äöüÄÖÜß]")
_DE_FUNCTION_WORDS = frozenset({
    "der", "die", "das", "und", "oder", "von", "zu", "mit", "bei", "für",
    "ein", "eine", "einer", "einem", "den", "dem", "des", "im", "am",
    "auf", "aus", "über", "unter", "nach", "vor", "zwischen",
    "als", "wie", "wenn", "weil",
})


def detect_term_language(term: str) -> str:
    """Guess the language of a search term.

    Returns an ISO language code, currently only 'de' or 'en'. The
    default is 'en' — safer than 'de', because scholarly and technical
    search terms are mostly English, even in German pipelines.

    Heuristic (in this order):
      1. contains umlauts or ß → 'de'
      2. at least one German function word as a whole token → 'de'
      3. otherwise → 'en'

    The function is deliberately simple — real language detection would
    be wrong here, because many terms are language-neutral ('DGX B300',
    'CERN', 'Jonas Brenner'). The heuristic only serves to split
    mixed-language term lists cleanly into two buckets, so that the
    connector can choose suitable search engines per language.
    """
    if not term or not isinstance(term, str):
        return "en"
    s = term.strip()
    if not s:
        return "en"
    if _DE_DIACRITICS.search(s):
        return "de"
    tokens = {t.lower() for t in re.findall(r"[\w'-]+", s) if len(t) >= 2}
    if tokens & _DE_FUNCTION_WORDS:
        return "de"
    return "en"


# ── Normalising the LLM output ────────────────────────────────


def _flatten_term(value) -> Optional[str]:
    """Turn a 'term' element into a clean string, or None."""
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, dict):
        # Some LLM answers nest further:
        # {"term": "...", "lang": "de"} or {"de": "..."}
        for key in ("term", "text", "query", "value"):
            if key in value and isinstance(value[key], str):
                return value[key].strip() or None
        # Last attempt: the only string value in the dict
        strings = [v for v in value.values() if isinstance(v, str)]
        if len(strings) == 1 and strings[0].strip():
            return strings[0].strip()
        return None
    # number, bool etc. → not useful as a search term
    return None


def coerce_search_terms(
    raw,
) -> tuple[list[str], dict[str, list[str]]]:
    """Normalise the LLM output for `search_terms` into two forms.

    Accepted input formats (seen from the analysis LLM in practice):

      a) list of strings:
           ["DGX B300 Stromaufnahme", "DGX B300 power consumption"]

      b) list of dicts with a language:
           [{"term": "...", "lang": "de"},
            {"term": "...", "lang": "en"}]

      c) dict with languages as keys:
           {"de": ["term1", "term2"], "en": ["term3"]}

      d) a single string:
           "DGX B300"

      e) None / empty:
           []

    Returns:
        (flat_terms, terms_by_lang)
        - flat_terms: all terms as a flat list, in their original order
        - terms_by_lang: dict[lang, list[str]], empty languages omitted
    """
    flat: list[str] = []
    by_lang: dict[str, list[str]] = {}

    def _add(term: str, lang: Optional[str]) -> None:
        term = term.strip()
        if not term:
            return
        if term in flat:
            return  # Dedup
        flat.append(term)
        lc = (lang or detect_term_language(term)).lower()
        # Restrict to known language codes — otherwise 'german', 'deu',
        # 'english' end up in chaotic buckets.
        if lc in ("de", "ger", "deu", "deutsch", "german"):
            lc = "de"
        elif lc in ("en", "eng", "english", "englisch"):
            lc = "en"
        by_lang.setdefault(lc, []).append(term)

    if raw is None:
        return flat, by_lang

    # Form (d): a single string
    if isinstance(raw, str):
        _add(raw, None)
        return flat, by_lang

    # Form (c): dict per language
    if isinstance(raw, dict):
        # Heuristic: do the keys look like language codes (short, alphabetic)?
        keys = list(raw.keys())
        looks_like_lang_dict = all(
            isinstance(k, str) and len(k) <= 10 and k.replace("-", "").isalpha()
            for k in keys
        )
        if looks_like_lang_dict:
            for lang, value in raw.items():
                if isinstance(value, str):
                    _add(value, lang)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        term = _flatten_term(item)
                        if term:
                            _add(term, lang)
            return flat, by_lang
        # Otherwise: a single dict with term/lang fields (form b without a list)
        term = _flatten_term(raw)
        lang = raw.get("lang") if isinstance(raw, dict) else None
        if term:
            _add(term, lang)
        return flat, by_lang

    # Form (a) or (b): list
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str):
                _add(item, None)
            elif isinstance(item, dict):
                term = _flatten_term(item)
                lang = item.get("lang") or item.get("language")
                if term:
                    _add(term, lang)
            # ignore other types
        return flat, by_lang

    # Unknown type — return empty rather than raising
    return flat, by_lang


# ── Aggregation over all questions ──────────────────────────────────


def build_terms_by_lang(
    questions: Iterable,
) -> tuple[dict[str, list[str]], set[str]]:
    """Collect all search terms of all questions by language.

    Prefers the already structured `search_terms_by_lang` of the
    `ResearchQuestion` object; falls back to heuristic detection over
    `search_terms` if the former is empty.

    The question's `search_langs` hints at which languages the LLM
    expected — we respect it: if the heuristic language is not in the
    question's `search_langs`, the first language from `search_langs`
    is used. That prevents an English-sounding term such as
    'Stromaufnahme DGX' from being pushed into the 'en' bucket when the
    question only searches in German.

    Args:
        questions: iterable of ResearchQuestion (duck-typed — only
                   .search_terms, .search_terms_by_lang and .search_langs
                   are accessed).

    Returns:
        (terms_by_lang, all_langs)
        - terms_by_lang: dict[lang, list[str]] — de-duplicated terms per
          language (not sorted; the caller sorts if needed).
        - all_langs: set[str] of all language codes used.
    """
    terms_by_lang: dict[str, list[str]] = {}
    all_langs: set[str] = set()

    for q in questions:
        # Preferred source: already structured
        structured = getattr(q, "search_terms_by_lang", None) or {}
        question_langs = list(getattr(q, "search_langs", []) or [])

        if structured and isinstance(structured, dict):
            for lang, terms in structured.items():
                if not isinstance(terms, (list, tuple)):
                    continue
                lc = lang.lower() if isinstance(lang, str) else "en"
                bucket = terms_by_lang.setdefault(lc, [])
                for t in terms:
                    if isinstance(t, str) and t.strip() and t not in bucket:
                        bucket.append(t.strip())
                all_langs.add(lc)
        else:
            # Fallback: flat list with the language heuristic
            flat = getattr(q, "search_terms", []) or []
            default_lang = (question_langs[0] if question_langs else "en").lower()
            for t in flat:
                if not isinstance(t, str) or not t.strip():
                    continue
                t = t.strip()
                detected = detect_term_language(t)
                # If the heuristic language is not in the question's search_langs,
                # take the first search_langs language as the default. That respects
                # what the LLM specified.
                if question_langs and detected not in question_langs:
                    lang = default_lang
                else:
                    lang = detected
                bucket = terms_by_lang.setdefault(lang, [])
                if t not in bucket:
                    bucket.append(t)
                all_langs.add(lang)

        # Make sure that all declared search_langs of the question appear in
        # all_langs — even if no terms ended up in them. That lets the
        # search-strategy report list all declared languages.
        for lang in question_langs:
            if isinstance(lang, str) and lang.strip():
                all_langs.add(lang.strip().lower())

    return terms_by_lang, all_langs
