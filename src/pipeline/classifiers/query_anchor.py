"""
Query anchor classifier.

Determines whether the research request has a person or organisation
anchor. The result controls whether the person hallucination filter is
activated at all.

This is an LLM decision rather than a heuristic on capitalised nouns: in
German every noun is capitalised, and a heuristic reads a request like
"Vergleich DGX B300 vs B200 — Preis-Leistungs-Verhältnis" as containing
the person "Preis-Leistungs-Verhältnis" — the filter then discards
every extract and the report ends up empty.

Usage:
    anchor = await classify_query_anchor(query, plan_summary, llm)
    if anchor.type == "person" and anchor.confidence >= 0.7:
        # filter active
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.core.limits import CLASSIFIER_CONFIDENCE_THRESHOLD
from src.llm.json_parser import coerce_float
from src.pipeline.classifiers.base import (
    ClassifierCall,
    ClassifierLLM,
    ClassifierResult,
    call_classifier,
)

logger = logging.getLogger(__name__)


# ─── Result data model ────────────────────────────────────────────


@dataclass
class QueryAnchor(ClassifierResult):
    """Result of `classify_query_anchor`.

    `type` is one of:
      - "person"       — the research is about one specific person
      - "organization" — about one specific organisation/company/institution
      - "none"         — no clear anchor (subject matter, comparison, general)

    With `confidence < 0.7`, `type` is forced to "none" (safe default).
    """
    type: str = "none"        # "person" | "organization" | "none"
    target: str = ""          # e.g. "Jonas Brenner" — empty with type="none"

    def is_person(self) -> bool:
        return (
            self.type == "person"
            and self.target.strip() != ""
            and self.confidence >= CLASSIFIER_CONFIDENCE_THRESHOLD
        )

    def is_organization(self) -> bool:
        return (
            self.type == "organization"
            and self.target.strip() != ""
            and self.confidence >= CLASSIFIER_CONFIDENCE_THRESHOLD
        )


# ─── Prompt ──────────────────────────────


QUERY_ANCHOR_PROMPT = """Determine whether the following research request has a clearly identifiable
PERSON or ORGANISATION ANCHOR.

A PERSON anchor exists if the research revolves around one specific
person and facts about this person are to be collected.
Examples of person anchors:
  "Jonas Brenner Forschung" → yes, target="Jonas Brenner"
  "Who is Prof. Müller at Example University?" → yes, target="Prof. Müller"
  "Lebenslauf von Dr. Schmidt" → yes, target="Dr. Schmidt"
  "Publications Mira Kovacs" → yes, target="Mira Kovacs"

An ORGANISATION anchor exists if the research revolves around one
specific organisation/company/institution and facts about this
organisation are to be collected.
Examples of organisation anchors:
  "Was macht das Fraunhofer-Institut für IIS?" → yes, target="Fraunhofer-Institut für IIS"
  "History of the Charité Berlin" → yes, target="Charité Berlin"

There is NO anchor for:
  "Vergleich DGX B300 vs B200 Preis-Leistungs-Verhältnis" → no
  "Current AI regulation in the EU" → no
  "Digitalisierung an Hochschulen" → no
  "Best open-source LLMs 2025" → no
  "Differences between LDAP and Active Directory" → no

Even if people or organisations appear on the side but are NOT the
main subject, it is NO anchor:
  "AI regulation in the US and the EU" → no anchor (US/EU are jurisdictions)
  "Studienangebot der Beispiel-Universität" → no anchor (the university is context, not the object of study)

Note: in German every noun is capitalised — capitalisation alone is no
sign of a name.

QUERY: {query}
PLAN SUMMARY: {plan_summary}

Answer as JSON:
{{
  "anchor_type": "person" | "organization" | "none",
  "target": "Jonas Brenner" or "" if none,
  "confidence": 0.0 to 1.0,
  "reasoning": "short, concrete justification — name the signals you weighed"
}}

With a confidence below 0.7: set anchor_type="none". Better no anchor
than a wrong one.
"""


# ─── Main function ─────────────────────────────────────────────────


async def classify_query_anchor(
    query: str,
    plan_summary: str,
    llm: ClassifierLLM,
) -> tuple[QueryAnchor, ClassifierCall]:
    """LLM classifier: does the query have a person/organisation anchor?

    Args:
        query: the user's original research request.
        plan_summary: summary from the research plan (additional context
            if the query is short).
        llm: LLM client (can be mocked in tests).

    Returns:
        (anchor, call) — anchor is the typed answer, call the logging
        object for structured logs.

    Conservative default on failure or low confidence:
        type="none", confidence=0.0 — the filter is NOT activated.
        Better a genuine person search without the filter than a
        subject-matter request destroyed by a wrong filter.
    """
    prompt = QUERY_ANCHOR_PROMPT.format(
        query=query.strip() or "(empty)",
        plan_summary=(plan_summary or "").strip() or "(none)",
    )

    parsed, call = await call_classifier(
        name="query_anchor",
        llm=llm,
        prompt=prompt,
        expected_keys=["anchor_type", "confidence"],
        max_tokens=512,
        input_summary={
            "query": query[:200],
            "plan_summary": (plan_summary or "")[:200],
        },
    )

    if not parsed:
        # Conservative default: no anchor
        anchor = QueryAnchor(
            type="none",
            target="",
            confidence=0.0,
            reasoning="LLM call or JSON parse failed — safe default 'none'",
            fallback_used=True,
        )
        call.output = {"anchor_type": "none", "confidence": 0.0,
                       "fallback": True}
        return anchor, call

    raw_type = str(parsed.get("anchor_type", "none")).lower().strip()
    raw_target = str(parsed.get("target", "") or "").strip()
    raw_conf = coerce_float(parsed.get("confidence", 0.0))
    raw_reason = str(parsed.get("reasoning", "") or "").strip()

    # Schema validation
    if raw_type not in {"person", "organization", "none"}:
        logger.warning(
            "query_anchor: invalid anchor_type %r — falling back to 'none'",
            raw_type,
        )
        raw_type = "none"

    # Consistency: no target → no anchor
    if raw_type in {"person", "organization"} and not raw_target:
        logger.warning(
            "query_anchor: type=%s, but target empty → forcing 'none'",
            raw_type,
        )
        raw_type = "none"

    # Consistency: type=none → empty target
    if raw_type == "none":
        raw_target = ""

    # Main rule: with confidence < 0.7, type="none" is forced.
    # This is not just a warning but the active safe default.
    # Better no filter than a wrong one.
    fallback = False
    if raw_type != "none" and raw_conf < CLASSIFIER_CONFIDENCE_THRESHOLD:
        logger.info(
            "query_anchor: confidence=%.2f < %.2f → type forced to 'none' "
            "(was %s, target=%r)",
            raw_conf, CLASSIFIER_CONFIDENCE_THRESHOLD, raw_type, raw_target,
        )
        raw_type = "none"
        raw_target = ""
        fallback = True
        raw_reason = (
            f"Confidence below the threshold ({raw_conf:.2f} < "
            f"{CLASSIFIER_CONFIDENCE_THRESHOLD:.2f}) — "
            f"safe default. Original reasoning: {raw_reason}"
        )

    anchor = QueryAnchor(
        type=raw_type,
        target=raw_target,
        confidence=raw_conf,
        reasoning=raw_reason,
        fallback_used=fallback,
    )

    return anchor, call
