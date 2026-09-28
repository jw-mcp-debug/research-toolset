"""
Coverage and continue-decision classifiers.

Whether a question is answered and whether another research round is
worthwhile are judged by an LLM on the content, not by mechanical
thresholds such as "at least 3 relevant extracts from 2 sources" or
"stop when a round yields nothing new". Such thresholds make no content
judgement: they can declare every question "unanswerable" after two
rounds when the real cause is a filter that was switched on wrongly.

Important: `evaluate_coverage` has its own answer class `filter_blocked`,
so that the difference between "the world has no data" and "a filter
swallowed everything" becomes visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.core.limits import CLASSIFIER_CONFIDENCE_FALLBACK
from src.llm.json_parser import coerce_float
from src.pipeline.classifiers.base import (
    ClassifierCall,
    ClassifierLLM,
    ClassifierResult,
    call_classifier,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# B.2 — evaluate_coverage
# ═══════════════════════════════════════════════════════════════════


VALID_COVERAGE = {"answered", "partial", "unanswered", "filter_blocked"}


@dataclass
class CoverageResult(ClassifierResult):
    """Result of `evaluate_coverage` for a single question."""
    coverage: str = "partial"  # answered | partial | unanswered | filter_blocked
    missing_aspects: list[str] = field(default_factory=list)
    supporting_extract_ids: list[str] = field(default_factory=list)


COVERAGE_PROMPT = """Assess whether the following research question is answered by the
extracts at hand.

QUESTION (ID {question_id}, priority {priority}):
{question}

EXTRACTS (all that concern this question):
{extracts_with_sources}

FILTER CONTEXT:
- Filters active in this round: {active_filters}
- Loss rate of these filters: {filter_loss_rate_pct}%

Assess the coverage:

- "answered": the question is essentially answered. Core aspects are
  supported by at least 2 independent sources OR by one strong
  primary source.
- "partial": some aspects answered, others open. Another round
  could help to search in a more targeted way.
- "unanswered": no relevant extracts. There can be two reasons —
  see filter_blocked.
- "filter_blocked": there were extracts from the sources, but the
  filter discarded them. Recognisable by a high filter loss rate in
  this round (> 70%) AND the question having no positive extracts despite
  many sources visited. IMPORTANT: if the filter loss rate is > 70%,
  prefer "filter_blocked" over "unanswered".

Answer as JSON:
{{
  "coverage": "answered" | "partial" | "unanswered" | "filter_blocked",
  "confidence": 0.0 to 1.0,
  "missing_aspects": ["concrete aspect 1", "aspect 2"],
  "supporting_extract_ids": ["E12", "E47"],
  "reasoning": "justification — name which aspects of the question are covered by which extracts"
}}

With a confidence below 0.5: prefer "partial" as the safe default —
researching further is better than stopping too early.
"""


def _format_extracts_for_coverage(extracts: list[dict]) -> str:
    """Format extracts for the prompt.

    Expects a list of dicts with the keys id, fact, source_title,
    reliability. SourceExtract objects (with the same attributes) are
    accepted as well — duck-typed.
    """
    if not extracts:
        return "(none)"
    out = []
    for i, e in enumerate(extracts, 1):
        if isinstance(e, dict):
            ext_id = e.get("id") or f"E{i}"
            fact = e.get("fact", "")
            source = e.get("source_title") or e.get("source_url", "?")
            rel = e.get("reliability", "?")
        else:
            ext_id = getattr(e, "id", None) or f"E{i}"
            fact = getattr(e, "fact", "")
            source = (getattr(e, "source_title", "")
                      or getattr(e, "source_url", "?"))
            rel = getattr(e, "reliability", "?")
        out.append(
            f"[{ext_id}] {fact}\n"
            f"     Source: {source}\n"
            f"     Reliability: {rel}"
        )
    return "\n\n".join(out)


async def evaluate_coverage(
    *,
    question_id: str,
    question: str,
    priority: str,
    extracts: list,                     # SourceExtract or dict
    active_filters: list[str],
    filter_loss_rate: float,            # 0.0 - 1.0
    llm: ClassifierLLM,
) -> tuple[CoverageResult, ClassifierCall]:
    """Assess the coverage of a single question.

    Rules:
      - with filter_loss_rate > 70 %, `filter_blocked` is preferred
      - with confidence < 0.5, `partial` is forced as the safe default

    Returns:
        (result, call)
    """
    extracts_text = _format_extracts_for_coverage(extracts)
    filters_text = ", ".join(active_filters) if active_filters else "(none)"
    loss_pct = round(filter_loss_rate * 100, 1)

    prompt = COVERAGE_PROMPT.format(
        question_id=question_id,
        priority=priority,
        question=question,
        extracts_with_sources=extracts_text,
        active_filters=filters_text,
        filter_loss_rate_pct=loss_pct,
    )

    parsed, call = await call_classifier(
        name="evaluate_coverage",
        llm=llm,
        prompt=prompt,
        expected_keys=["coverage", "confidence"],
        max_tokens=768,
        input_summary={
            "question_id": question_id,
            "priority": priority,
            "n_extracts": len(extracts),
            "active_filters": active_filters,
            "filter_loss_rate": filter_loss_rate,
        },
    )

    if not parsed:
        return CoverageResult(
            coverage="partial",
            confidence=0.0,
            reasoning="LLM fallback — safe default 'partial'",
            fallback_used=True,
        ), call

    raw_cov = str(parsed.get("coverage", "partial")).lower().strip()
    if raw_cov not in VALID_COVERAGE:
        logger.warning("evaluate_coverage: invalid coverage %r → 'partial'",
                       raw_cov)
        raw_cov = "partial"

    raw_conf = coerce_float(parsed.get("confidence", 0.0))

    # Conservative default at low confidence: "partial"
    fallback = False
    if raw_conf < CLASSIFIER_CONFIDENCE_FALLBACK and raw_cov != "partial":
        logger.info(
            "evaluate_coverage: confidence=%.2f < %.2f → coverage forced "
            "to 'partial' (was %s)",
            raw_conf, CLASSIFIER_CONFIDENCE_FALLBACK, raw_cov,
        )
        raw_cov = "partial"
        fallback = True

    missing = parsed.get("missing_aspects", []) or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    missing = [str(m).strip() for m in missing if str(m).strip()]

    supp_ids = parsed.get("supporting_extract_ids", []) or []
    if not isinstance(supp_ids, list):
        supp_ids = [str(supp_ids)]
    supp_ids = [str(s).strip() for s in supp_ids if str(s).strip()]

    reasoning = str(parsed.get("reasoning", "") or "").strip()

    # Consistency note: with coverage="answered", missing_aspects should be empty
    if raw_cov == "answered" and missing:
        logger.debug(
            "evaluate_coverage: coverage=answered, but missing_aspects=%r — "
            "consistency warning (not an error, just a notice)", missing,
        )

    return CoverageResult(
        coverage=raw_cov,
        confidence=raw_conf,
        missing_aspects=missing,
        supporting_extract_ids=supp_ids,
        reasoning=reasoning,
        fallback_used=fallback,
    ), call


# ═══════════════════════════════════════════════════════════════════
# B.3 — decide_continue_research
# ═══════════════════════════════════════════════════════════════════


VALID_DECISIONS = {
    "continue", "stop_done", "stop_diminishing", "stop_filter_problem",
}


@dataclass
class ContinueDecision(ClassifierResult):
    """Result of `decide_continue_research`."""
    decision: str = "continue"
    next_round_focus: str = ""

    def should_continue(self) -> bool:
        return self.decision == "continue"


CONTINUE_PROMPT = """You control an autonomous research run. Decide whether another round
is worthwhile.

CURRENT STATE after round {round_number} of at most {max_rounds}:
- {answered_count} of {total_count} questions answered
- {partial_count} questions partially answered
- {unanswered_count} questions not answered
- {filter_blocked_count} questions blocked by the filter
- This round: {new_extracts} new positive extracts, {new_sources} new sources
- Filter loss rate this round: {filter_loss_rate_pct}%
- So far in total: {total_extracts} extracts from {total_sources} sources

COVERAGE DETAILS (from evaluate_coverage):
{coverage_details}

Decision options:

- "continue": another round is worthwhile. This is the right choice if:
  - there are still clear partial/unanswered questions
  - the last round still brought substantial new extracts
  - the filter apparently blocks a lot and should be reviewed

- "stop_done": the research is complete. Choose if:
  - all questions are "answered" OR
  - the most important questions (priority "high") are "answered" and the
    rest realistically cannot be clarified further

- "stop_diminishing": further rounds are unlikely to help. Choose if:
  - the last 2 rounds brought hardly any new positive extracts
  - the coverage no longer changes substantially

- "stop_filter_problem": stop the research because of a recognisable filter
  misconfiguration. Choose if:
  - many questions are "filter_blocked"
  - the filter loss rate was > 70% in several rounds
  - further rounds would probably come to nothing in the same way

Answer as JSON:
{{
  "decision": "continue" | "stop_done" | "stop_diminishing" | "stop_filter_problem",
  "confidence": 0.0 to 1.0,
  "next_round_focus": "if 'continue': a concrete recommendation of what the next round should prioritise, e.g. 'focus on F3 and F5 with broader search terms'",
  "reasoning": "justification"
}}

With a confidence below 0.5 and round_number < max_rounds: choose "continue"
as the safe default.
"""


async def decide_continue_research(
    *,
    round_number: int,
    max_rounds: int,
    answered_count: int,
    partial_count: int,
    unanswered_count: int,
    filter_blocked_count: int,
    total_count: int,
    new_extracts: int,
    new_sources: int,
    filter_loss_rate: float,
    total_extracts: int,
    total_sources: int,
    coverage_details: str,
    llm: ClassifierLLM,
) -> tuple[ContinueDecision, ClassifierCall]:
    """Decide whether another research round is worthwhile.

    Rules:
      - next_round_focus must be set with decision="continue"
      - with confidence < 0.5 and round_number < max_rounds → "continue"
        as the safe default
    """
    prompt = CONTINUE_PROMPT.format(
        round_number=round_number,
        max_rounds=max_rounds,
        answered_count=answered_count,
        partial_count=partial_count,
        unanswered_count=unanswered_count,
        filter_blocked_count=filter_blocked_count,
        total_count=total_count,
        new_extracts=new_extracts,
        new_sources=new_sources,
        filter_loss_rate_pct=round(filter_loss_rate * 100, 1),
        total_extracts=total_extracts,
        total_sources=total_sources,
        coverage_details=coverage_details or "(no details available)",
    )

    parsed, call = await call_classifier(
        name="decide_continue_research",
        llm=llm,
        prompt=prompt,
        expected_keys=["decision", "confidence"],
        max_tokens=768,
        input_summary={
            "round_number": round_number,
            "max_rounds": max_rounds,
            "answered_count": answered_count,
            "partial_count": partial_count,
            "unanswered_count": unanswered_count,
            "filter_blocked_count": filter_blocked_count,
            "filter_loss_rate": filter_loss_rate,
        },
    )

    if not parsed:
        # Conservative default: keep researching while rounds are left
        if round_number < max_rounds:
            return ContinueDecision(
                decision="continue",
                confidence=0.0,
                next_round_focus="LLM fallback — continue with the default strategy",
                reasoning="LLM fallback — safe default 'continue'",
                fallback_used=True,
            ), call
        else:
            return ContinueDecision(
                decision="stop_done",
                confidence=0.0,
                reasoning="LLM fallback — max_rounds reached",
                fallback_used=True,
            ), call

    raw_dec = str(parsed.get("decision", "continue")).lower().strip()
    if raw_dec not in VALID_DECISIONS:
        logger.warning("decide_continue_research: invalid decision %r → 'continue'",
                       raw_dec)
        raw_dec = "continue"

    raw_conf = coerce_float(parsed.get("confidence", 0.0))

    # Conservative default at low confidence while rounds are still available
    fallback = False
    if (raw_conf < CLASSIFIER_CONFIDENCE_FALLBACK
            and round_number < max_rounds
            and raw_dec != "continue"):
        logger.info(
            "decide_continue_research: confidence=%.2f < %.2f and round %d/%d "
            "→ decision forced to 'continue' (was %s)",
            raw_conf, CLASSIFIER_CONFIDENCE_FALLBACK,
            round_number, max_rounds, raw_dec,
        )
        raw_dec = "continue"
        fallback = True

    focus = str(parsed.get("next_round_focus", "") or "").strip()
    reasoning = str(parsed.get("reasoning", "") or "").strip()

    # Consistency: with "continue" a focus should be given
    if raw_dec == "continue" and not focus:
        focus = "No specific instruction — default strategy"

    return ContinueDecision(
        decision=raw_dec,
        confidence=raw_conf,
        next_round_focus=focus,
        reasoning=reasoning,
        fallback_used=fallback,
    ), call
