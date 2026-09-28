"""
Follow-up search queries for the next research round.

After each round the coverage classifier states, per research question,
whether it is answered and which aspects are still missing. When the
continue decision asks for another round, this module turns that assessment
into search queries for the next round — deterministically, without another
LLM call:

  - questions assessed as "partial" or "unanswered" get one query per
    missing aspect (at most MAX_ASPECTS_PER_QUESTION), anchored with the
    question's own first search term in each of its languages;
  - an "unanswered" question without named missing aspects is searched again
    with the question text itself;
  - "answered" questions get nothing, and neither do "filter_blocked" ones:
    their sources were found but filtered out, so more searching does not
    help (the diagnosis reports this case).

The result uses the follow-up query format that `_run_search_and_fetch`
reads for rounds after the first: {"question_id", "search_terms": {lang:
[terms]}, "search_langs", "reason"}.
"""

from __future__ import annotations

import re

MAX_ASPECTS_PER_QUESTION = 3
MAX_WORDS_PER_QUERY = 10
NEEDS_MORE = ("partial", "unanswered")


def _clean(text: str) -> str:
    """One line, no list markers or trailing punctuation."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text.strip(" -*•.;:,\"'")


def _shorten(words: str) -> str:
    parts = words.split()
    return " ".join(parts[:MAX_WORDS_PER_QUERY])


def _anchors(question) -> dict[str, str]:
    """First search term per language of a question (the topic anchor)."""
    by_lang = getattr(question, "search_terms_by_lang", None) or {}
    anchors = {
        lang: _clean(terms[0])
        for lang, terms in by_lang.items()
        if isinstance(terms, list) and terms and _clean(terms[0])
    }
    if not anchors:
        terms = getattr(question, "search_terms", None) or []
        langs = getattr(question, "search_langs", None) or ["en"]
        if terms and _clean(terms[0]):
            anchors = {lang: _clean(terms[0]) for lang in langs}
    return anchors


def build_followup_queries(plan, coverage_results: dict) -> list[dict]:
    """Follow-up queries for all questions the coverage assessment left open.

    Args:
        plan: the ResearchPlan (duck-typed: `.questions` with `.id`,
            `.question`, `.search_terms`, `.search_terms_by_lang`,
            `.search_langs`).
        coverage_results: {question_id: {"coverage": ..., "missing_aspects":
            [...]}} as stored by CoverageNode for the last round.

    Returns:
        List of follow-up query dicts; empty if nothing is left to search.
    """
    followups: list[dict] = []
    for q in getattr(plan, "questions", None) or []:
        result = (coverage_results or {}).get(q.id) or {}
        coverage = result.get("coverage")
        if coverage not in NEEDS_MORE:
            continue
        anchors = _anchors(q)
        langs = list(anchors) or list(getattr(q, "search_langs", None) or ["en"])
        aspects = [_clean(a) for a in (result.get("missing_aspects") or [])]
        aspects = [a for a in aspects if a][:MAX_ASPECTS_PER_QUESTION]

        terms: dict[str, list[str]] = {}
        if aspects:
            for lang in langs:
                anchor = anchors.get(lang, "")
                for aspect in aspects:
                    query = _shorten(f"{anchor} {aspect}".strip())
                    if query and query not in terms.setdefault(lang, []):
                        terms[lang].append(query)
            reason = f"{coverage}: missing aspects of {q.id}"
        elif coverage == "unanswered" and _clean(q.question):
            text = _shorten(_clean(q.question))
            terms = {lang: [text] for lang in langs}
            reason = f"unanswered: {q.id} searched again with the question itself"
        else:
            continue

        if any(terms.values()):
            followups.append({
                "question_id": q.id,
                "search_terms": terms,
                "search_langs": langs,
                "reason": reason,
            })
    return followups
