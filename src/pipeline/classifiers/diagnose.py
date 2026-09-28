"""
Pipeline diagnosis classifier.

At the end of a research run, diagnoses why the pipeline ended the way
it did. A run that ends with an empty report because filters discarded
everything thus gets an explicit diagnosis `filter_too_strict` with a
reason and a suggested remedy, instead of just "empty report".

The diagnosis is:
  - shown prominently as a banner in the report
  - linked in the UI to a "restart without filter X" action
  - written to the structured log
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.output_language import current as current_language, prompt_language_line, tc
from src.core.limits import (
    FILTER_LOSS_RATE_WARNING,
)
from src.llm.json_parser import coerce_float
from src.pipeline.classifiers.base import (
    ClassifierCall,
    ClassifierLLM,
    ClassifierResult,
    call_classifier,
)

logger = logging.getLogger(__name__)


VALID_DIAGNOSES = {
    "successful",
    "data_scarcity",
    "filter_too_strict",
    "wrong_query",
    "wrong_sources",
    "partial_success",
}

# Diagnoses that mean "no success" — the UI highlights them.
PROBLEMATIC_DIAGNOSES = {
    "filter_too_strict",
    "wrong_query",
    "wrong_sources",
}


@dataclass
class DiagnosisResult(ClassifierResult):
    """Result of `diagnose_pipeline_state`."""
    diagnosis: str = "partial_success"
    remediation: str = ""
    user_message: str = ""

    def is_successful(self) -> bool:
        return self.diagnosis == "successful"

    def is_problematic(self) -> bool:
        return self.diagnosis in PROBLEMATIC_DIAGNOSES


DIAGNOSE_PROMPT = """Diagnose the state of this research run before the report
is written. This diagnosis becomes visible in the report and in the UI.

FINAL STATE:
- Query: {query}
- Use case: {use_case}
- Rounds: {n_rounds} (of at most {max_rounds})
- Stop reason: {stop_reason}
- Total extracts: {final_extracts}
- Of which positive: {positive_extracts}
- Of which negative (the source mentions the question but gives no answer): {negative_extracts}
- Of which meta (statements about the source situation): {meta_extracts}
- Total sources: {n_sources}
- Coverage distribution of the questions:
  - answered: {coverage_answered}
  - partial: {coverage_partial}
  - unanswered: {coverage_unanswered}
  - filter_blocked: {coverage_filter_blocked}

FILTER STATISTICS:
{filter_stats}

CLASSIFIER ACTIVATIONS:
- query_anchor: {anchor_type} (confidence {anchor_confidence:.2f})
- active filters: {active_filters}

Diagnosis options:

- "successful": the research reached its goal. Most questions
  answered, filters had a sensible effect (loss rate < 50%),
  solid source situation.

- "data_scarcity": the world has little publicly accessible
  information on this topic. Recognisable by: many unanswered (not
  filter_blocked!), few sources found per question, many sources
  sorted out as irrelevant.

- "filter_too_strict": filters removed too much. Recognisable
  by a high filter loss rate (> 70%) and many filter_blocked questions.
  A clear symptom of a wrongly activated filter.

- "wrong_query": the request itself is too vague or contains typos/
  unknown terms. Recognisable by: SearXNG found few hits OR
  the hits were systematically off topic.

- "wrong_sources": the accessible sources are of the wrong types.
  Recognisable by: many hits, but mostly listing pages/news instead of
  primary sources. The request would need e.g. more PDFs/datasheets.

- "partial_success": some questions answered, others not — but
  no clear pipeline problem.

Answer as JSON:
{{
  "diagnosis": "successful" | "data_scarcity" | "filter_too_strict" |
               "wrong_query" | "wrong_sources" | "partial_success",
  "confidence": 0.0 to 1.0,
  "remediation": "concrete recommendation for the user: what they could do",
  "user_message": "short, clear explanation for the user, in plain (non-technical) language",
  "reasoning": "technical justification referring to the statistics"
}}

Write user_message and remediation for the reader, in the output language
stated at the end of this prompt; reasoning may stay in English.
"""


def _format_filter_stats(filter_stats: dict) -> str:
    """Format the filter_stats dict for the prompt.

    Expected format:
        {"person_hallucination": {"activated": False, "rejected": 0, "total": 234},
         "negative_extract": {"activated": True, "rejected": 12, "total": 234},
         ...}
    """
    if not filter_stats:
        return "(no filter statistics)"
    lines = []
    for name, stats in filter_stats.items():
        if not isinstance(stats, dict):
            continue
        activated = stats.get("activated", False)
        rejected = stats.get("rejected", 0)
        total = stats.get("total", 0)
        rate = (rejected / total * 100) if total > 0 else 0.0
        if activated:
            lines.append(
                f"  - {name}: active, {rejected}/{total} discarded ({rate:.1f}%)"
            )
        else:
            lines.append(f"  - {name}: not active")
    return "\n".join(lines) if lines else "(no filter statistics)"


async def diagnose_pipeline_state(
    *,
    query: str,
    use_case: str,
    n_rounds: int,
    max_rounds: int,
    stop_reason: str,
    final_extracts: int,
    positive_extracts: int,
    negative_extracts: int,
    meta_extracts: int,
    n_sources: int,
    coverage_distribution: dict,         # {answered, partial, unanswered, filter_blocked}
    filter_stats: dict,
    anchor_type: str,
    anchor_confidence: float,
    active_filters: list[str],
    llm: ClassifierLLM,
) -> tuple[DiagnosisResult, ClassifierCall]:
    """Diagnose the pipeline state at the end of a run.

    Returns:
        (result, call) — on fallback `partial_success` is returned with
        confidence 0 (the user should judge the result themselves).
    """
    cd = coverage_distribution or {}
    prompt = DIAGNOSE_PROMPT.format(
        query=query[:500],
        use_case=use_case or "(unknown)",
        n_rounds=n_rounds,
        max_rounds=max_rounds,
        stop_reason=stop_reason or "(unknown)",
        final_extracts=final_extracts,
        positive_extracts=positive_extracts,
        negative_extracts=negative_extracts,
        meta_extracts=meta_extracts,
        n_sources=n_sources,
        coverage_answered=cd.get("answered", 0),
        coverage_partial=cd.get("partial", 0),
        coverage_unanswered=cd.get("unanswered", 0),
        coverage_filter_blocked=cd.get("filter_blocked", 0),
        filter_stats=_format_filter_stats(filter_stats),
        anchor_type=anchor_type or "none",
        anchor_confidence=anchor_confidence or 0.0,
        active_filters=", ".join(active_filters) if active_filters else "(none)",
    )
    prompt += prompt_language_line(current_language())

    parsed, call = await call_classifier(
        name="diagnose_pipeline_state",
        llm=llm,
        prompt=prompt,
        expected_keys=["diagnosis", "confidence"],
        max_tokens=1024,
        input_summary={
            "n_rounds": n_rounds,
            "n_sources": n_sources,
            "final_extracts": final_extracts,
            "coverage_distribution": dict(cd),
            "active_filters": active_filters,
        },
    )

    if not parsed:
        return DiagnosisResult(
            diagnosis="partial_success",
            confidence=0.0,
            remediation=tc("diagnosis.fallback_remediation"),
            user_message=(
                tc("diagnosis.fallback_message")
            ),
            reasoning="LLM-Fallback",
            fallback_used=True,
        ), call

    raw_diag = str(parsed.get("diagnosis", "partial_success")).lower().strip()
    if raw_diag not in VALID_DIAGNOSES:
        logger.warning("diagnose: invalid diagnosis %r → 'partial_success'",
                       raw_diag)
        raw_diag = "partial_success"

    raw_conf = coerce_float(parsed.get("confidence", 0.0))
    remediation = str(parsed.get("remediation", "") or "").strip()
    user_message = str(parsed.get("user_message", "") or "").strip()
    reasoning = str(parsed.get("reasoning", "") or "").strip()

    # Schema consistency check:
    # - diagnosis="filter_too_strict" requires a filter loss rate >= WARNING
    # - diagnosis="successful" requires most questions to be "answered"
    fallback = False

    if raw_diag == "filter_too_strict":
        max_loss = max(
            (s.get("rejected", 0) / max(s.get("total", 1), 1)
             for s in filter_stats.values()
             if isinstance(s, dict) and s.get("activated", False)),
            default=0.0,
        )
        if max_loss < FILTER_LOSS_RATE_WARNING:
            logger.warning(
                "diagnose: filter_too_strict but max_loss=%.1f%% < %.0f%% — "
                "consistency error, falling back to partial_success",
                max_loss * 100, FILTER_LOSS_RATE_WARNING * 100,
            )
            raw_diag = "partial_success"
            fallback = True

    if raw_diag == "successful":
        total_q = sum(cd.get(k, 0) for k in
                      ("answered", "partial", "unanswered", "filter_blocked"))
        if total_q > 0 and cd.get("answered", 0) <= total_q / 2:
            logger.warning(
                "diagnose: successful but only %d/%d answered — "
                "consistency error, falling back to partial_success",
                cd.get("answered", 0), total_q,
            )
            raw_diag = "partial_success"
            fallback = True

    return DiagnosisResult(
        diagnosis=raw_diag,
        confidence=raw_conf,
        remediation=remediation or tc("diagnosis.no_remediation"),
        user_message=user_message or tc("diagnosis.done"),
        reasoning=reasoning,
        fallback_used=fallback,
    ), call
