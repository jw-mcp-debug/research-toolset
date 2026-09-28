"""
Search scope classifier (controls the search fan-out).

Without it, every search term would get up to three SearXNG passes —
standard, `recent` (time_range=year) and `science` (categories=science) —
tripling the number of tasks even for requests where neither recency nor
scholarly engines matter.

This classifier decides BEFOREHAND which additional passes are useful.
It is deliberately a SEPARATE, narrow classifier rather than an extra
field on `query_anchor`, whose output switches the person filter and
should stay focused on that decision; one additional LLM call is
unproblematic with the available infrastructure.

FAIL-OPEN default: on error or low confidence BOTH additional passes are
kept. Searches are only saved when the classifier is sure the request is
"not time-critical / not scholarly". Better a few more searches than a
gap.

Usage:
    scope, call = await classify_search_scope(query, plan_summary, llm)
    if scope.time_sensitive: ...   # enable the recent pass
    if scope.academic: ...         # enable the science pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.llm.json_parser import coerce_float
from src.pipeline.classifiers.base import (
    ClassifierCall,
    ClassifierLLM,
    ClassifierResult,
    call_classifier,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchScope(ClassifierResult):
    """Result of `classify_search_scope`.

    time_sensitive: the answer depends on current/recent developments
                    (news, releases, prices, "current", "latest").
    academic:       the question benefits from scholarly engines
                    (Scholar/arXiv/PubMed/Semantic Scholar).
    languages:      ISO codes of the languages in which relevant material
                    probably exists. EMPTY = no statement → the caller
                    uses the plan languages unchanged (fail-open).
                    Example: AI/ML topics → de/en/zh; a purely German
                    university topic → only de.
    recency:        with time_sensitive: how fresh must it be?
                    "" (= default year), "month" or "week" for real
                    breaking news.

    The default (also on fallback) is conservative: time_sensitive and
    academic True, languages empty (no language restriction), recency
    "" — so that NOTHING is restricted without a confident statement.
    """
    time_sensitive: bool = True
    academic: bool = True
    languages: list = None          # list[str] | None
    recency: str = ""               # "" | "year" | "month" | "week"

    def __post_init__(self):
        if self.languages is None:
            self.languages = []


SEARCH_SCOPE_PROMPT = """You classify a research request in order to steer the web search.

REQUEST:
{query}

PLAN SUMMARY:
{plan_summary}

Decide:

1. time_sensitive: does a good answer depend on CURRENT or recent
   developments? Yes for: news situation, new releases/versions,
   prices, market data, "current", "latest", "2025/2026", ongoing
   events, people in current roles. No for: timeless
   factual/conceptual questions, historical topics, definitions, methodology.

2. recency: ONLY if time_sensitive — how fresh? "week" for real
   breaking news/ongoing events, "month" for recent developments,
   "year" (default) for generally current topics. Otherwise "".

3. academic: does the answer benefit from SCHOLARLY sources
   (Google Scholar, arXiv, PubMed, Semantic Scholar)? Yes for:
   research, state of studies, theory, methods, technical terms, "state
   of research", depth in medicine/engineering. No for: everyday questions,
   product comparisons without a research angle, information about
   organisations/people, how-to.

4. languages: in which languages does the BEST material on THIS topic
   probably exist? ISO codes (de, en, zh, fr, es, ...).
   Guidelines:
   - International tech/AI/ML/research topics: ["en", "zh"] plus the
     language of the request (much high-quality AI literature is in
     English AND Chinese).
   - General international topics: ["en"] plus the language of the request.
   - Purely national/local/institution-specific topics (national law,
     people at one institution, national administration): only the language
     of that country — NO other languages, that would be a wasted search.
   - Topics with a clear country reference: the national language + en.
   Only give languages in which REALLY relevant material can be
   expected. When in doubt: English plus the language of the request.

Be rather conservative for 1/3 (when in doubt true). For 4 be deliberately
sparing — unnecessary languages multiply the search load.

Answer ONLY with JSON:
{{
  "time_sensitive": true|false,
  "recency": ""|"year"|"month"|"week",
  "academic": true|false,
  "languages": ["en"],
  "confidence": 0.0-1.0,
  "reasoning": "short justification"
}}"""


def _coerce_bool(v, default: bool = True) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "ja", "yes", "1"}
    if isinstance(v, (int, float)):
        return bool(v)
    return default


async def classify_search_scope(
    query: str,
    plan_summary: str,
    llm: ClassifierLLM,
) -> tuple[SearchScope, ClassifierCall]:
    """LLM classifier that controls the search fan-out."""
    prompt = SEARCH_SCOPE_PROMPT.format(
        query=(query or "").strip() or "(empty)",
        plan_summary=(plan_summary or "").strip() or "(none)",
    )

    parsed, call = await call_classifier(
        name="search_scope",
        llm=llm,
        prompt=prompt,
        expected_keys=["time_sensitive", "academic"],
        max_tokens=400,
        input_summary={"query": (query or "")[:200]},
    )

    if not parsed:
        scope = SearchScope(
            time_sensitive=True,
            academic=True,
            confidence=0.0,
            reasoning="LLM/JSON failed — conservative default "
                      "(both additional passes active)",
            fallback_used=True,
        )
        call.output = {"time_sensitive": True, "academic": True,
                       "fallback": True}
        return scope, call

    ts = _coerce_bool(parsed.get("time_sensitive", True), True)
    ac = _coerce_bool(parsed.get("academic", True), True)
    conf = coerce_float(parsed.get("confidence", 0.0))
    reason = str(parsed.get("reasoning", "") or "").strip()

    # Languages: normalise defensively — only short ISO-like codes,
    # de-duplicated, lower-case. Garbage/empty ⇒ [] (= no restriction).
    raw_langs = parsed.get("languages", [])
    langs: list[str] = []
    if isinstance(raw_langs, list):
        for x in raw_langs:
            c = str(x or "").strip().lower()
            if 2 <= len(c) <= 5 and c.replace("-", "").isalpha():
                if c not in langs:
                    langs.append(c)

    # recency: only allowed values, otherwise "" (= default year behaviour)
    rec = str(parsed.get("recency", "") or "").strip().lower()
    if rec not in {"year", "month", "week"}:
        rec = ""

    scope = SearchScope(
        time_sensitive=ts,
        academic=ac,
        languages=langs,
        recency=rec,
        confidence=conf,
        reasoning=reason,
        fallback_used=False,
    )
    call.output = {
        "time_sensitive": ts, "academic": ac,
        "languages": langs, "recency": rec, "confidence": conf,
    }
    return scope, call
