"""
Report quality check.

Works on a user request, a plan and collected extracts that are
synthesised into a report:
  - the raw material is the extracts
  - the planned sections are the research questions from the plan
  - the hallucination check compares report statements with extracts

Two functions:

  `check_report_quality(...)` — macroscopic quality check:
    completeness, consistency, hallucinations, redundancy.
    Segmented for long reports (along Markdown headings). Complements
    the factoid verifier, which checks atomically, with an overall
    assessment.

  `check_query_fulfillment(...)` — fulfilment check at the level of the
    original user query. Coverage checks per plan question; fulfilment
    checks whether the report answers the original request as a whole.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.llm.json_parser import parse_llm_json

if TYPE_CHECKING:
    from src.llm.classifier_adapter import HarvestModelAdapter
    from src.pipeline.models import (
        ResearchPlan,
    )

logger = logging.getLogger(__name__)


# ── Thresholds ────────────────────────────────────────────────


# Above this length the report is checked in segments.
QUALITY_FULLTEXT_THRESHOLD = 15_000

# Maximum length of the summarised extracts in the prompt.
EXTRACTS_MAX_FOR_QUALITY = 30_000

# Maximum excerpt length for the fulfilment check
FULFILLMENT_REPORT_MAX = 4_000


# ── Result data types ───────────────────────────────────────────



# Models occasionally answer with German keys or values (the contract used to
# be German); they are mapped to the English contract before parsing.
_KEY_ALIASES = {
    "bestanden": "passed", "gesamtbewertung": "rating", "maengel": "issues",
    "beschreibung": "description", "korrekturvorschlag": "suggestion",
    "frage_id": "question_id", "segmente": "segments", "art": "kind",
    "erfuellt": "fulfilled", "bewertung": "assessment", "nacharbeit": "rework",
}
_VALUE_ALIASES = {
    "gut": "good", "verbesserungswuerdig": "needs_improvement",
    "verbesserungswürdig": "needs_improvement", "mangelhaft": "poor",
    "fehlend": "missing", "inkonsistent": "inconsistent", "unklar": "unclear",
    "halluziniert": "hallucinated",
}


def _english_contract(obj):
    """Map German keys/enum values of an LLM answer to the English contract."""
    if isinstance(obj, dict):
        return {_KEY_ALIASES.get(k, k): _english_contract(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_english_contract(v) for v in obj]
    if isinstance(obj, str):
        return _VALUE_ALIASES.get(obj, obj)
    return obj

@dataclass
class QualityIssue:
    """A single complaint from the quality check."""
    kind: str = "unclear"
    # allowed values (part of the prompt's JSON contract):
    #   "missing", "inconsistent", "redundant", "unclear", "hallucinated"
    description: str = ""
    suggestion: str = ""
    # Which question does it concern? Often mappable for research reports
    question_id: str = ""
    # If checked in segments: which segments?
    segments: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "QualityIssue":
        """Read a QualityIssue from a dict (LLM JSON output).

        Defensive: a non-dict yields an empty instance. The `segment`
        field (singular) of the LLM output is converted into `segments`
        (list).
        """
        if not isinstance(d, dict):
            return cls()
        seg = d.get("segments", d.get("segment"))
        if isinstance(seg, list):
            segments = [
                int(s) for s in seg
                if isinstance(s, (int, str)) and str(s).strip().lstrip('-').isdigit()
            ]
        elif seg is not None:
            try:
                segments = [int(seg)]
            except (ValueError, TypeError):
                segments = []
        else:
            segments = []
        return cls(
            kind=str(d.get("kind", "unclear") or "unclear"),
            description=str(d.get("description", "") or ""),
            suggestion=str(d.get("suggestion", "") or ""),
            question_id=str(d.get("question_id", "") or ""),
            segments=segments,
        )


@dataclass
class QualityCheckResult:
    """Result of the report quality check."""
    passed: bool = True
    rating: str = ""    # "good" | "needs_improvement" | "poor"
    issues: list[QualityIssue] = field(default_factory=list)
    segments: int = 1            # how many segments were checked?
    fallback_used: bool = False  # LLM failure → conservative default


@dataclass
class FulfillmentResult:
    """Result of the fulfilment check."""
    fulfilled: bool = True
    assessment: str = ""
    rework: str = ""    # concrete suggestion of what is still missing
    fallback_used: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "FulfillmentResult":
        """Read a FulfillmentResult from a dict (LLM JSON).

        Defensive: a non-dict yields `fulfilled=True, fallback_used=True`
        (conservative default — with broken LLM output we assume the
        report is fine, rather than wrongly triggering rework).
        """
        if not isinstance(d, dict):
            return cls(fulfilled=True, fallback_used=True)
        return cls(
            fulfilled=bool(d.get("fulfilled", True)),
            assessment=str(d.get("assessment", "") or ""),
            rework=str(d.get("rework", "") or ""),
        )


# ── Prompts (adapted to the research context) ──────────────────────


_QUALITY_PROMPT = """You are checking a research report against the collected \
source extracts.

RESEARCH QUESTIONS FROM THE PLAN:
{plan_questions}

REPORT:
{report}

SOURCE EXTRACTS (factual basis from the sources):
--- BEGIN EXTRACTS ---
{extracts}
--- END EXTRACTS ---

Check systematically:
1. Are all research questions covered in the report?
2. Are there contradictions WITHIN the report?
3. Are there redundancies (the same statement several times)?
4. Is important information from the extracts missing?
5. HALLUCINATION CHECK: does the report contain names, figures, dates or \
facts that do NOT appear in the extracts? Every such place \
is an issue of the kind "hallucinated".

Answer as JSON (keys and enum values exactly as shown, in English):
{{
  "passed": true|false,
  "rating": "good"|"needs_improvement"|"poor",
  "issues": [
    {{
      "question_id": "F1"|"",
      "kind": "missing"|"inconsistent"|"redundant"|"unclear"|"hallucinated",
      "description": "short description of the issue",
      "suggestion": "concrete suggestion (or empty)"
    }}
  ]
}}
If there are no issues: an empty list, passed=true.
Write description and suggestion in the language of the report."""


_FULFILLMENT_PROMPT = """Check whether the following research report fulfils the \
user's original request in substance.

USER REQUEST:
"{query}"

REPORT EXCERPT:
{report_excerpt}

Assess:
1. Was the request answered as a whole?
2. Are there obviously missing aspects?

Answer as JSON (keys exactly as shown):
{{
  "fulfilled": true|false,
  "assessment": "short justification (1-2 sentences)",
  "rework": "concrete suggestion of what is still missing (or empty if fulfilled)"
}}
Write assessment and rework in the language of the report."""


# ── Quality check ───────────────────────────────────


async def check_report_quality(
    report: str,
    extracts: list,
    plan: Optional["ResearchPlan"],
    dual_llm,
) -> QualityCheckResult:
    """Check the report for completeness, consistency and hallucinations.

    For short reports (≤ QUALITY_FULLTEXT_THRESHOLD): full-text check.
    For longer reports: segment-wise check along Markdown headings; the
    findings are consolidated.

    Args:
        report: the final report text.
        extracts: list of SourceExtract objects (the positive ones are
                  used as the factual basis).
        plan: the research plan (for question context).
        dual_llm: DualLLMClient. Wrapped internally in a
                  HarvestModelAdapter.

    Returns:
        QualityCheckResult. For an empty report or on error:
        conservatively passed=True with fallback_used=True.
    """
    if not report or not report.strip():
        return QualityCheckResult(passed=True, fallback_used=True)

    from src.llm.classifier_adapter import HarvestModelAdapter
    llm = HarvestModelAdapter(dual_llm)

    # Context of the plan questions
    if plan and plan.questions:
        plan_questions_text = "\n".join(
            f"- [{q.id}] {q.question} ({q.priority})"
            for q in plan.questions
        )
    else:
        plan_questions_text = "(no plan available)"

    # Prepare the extracts — only positive ones (as in the factoid verifier).
    # Negative/meta extracts do not support report statements.
    extracts_text = _format_extracts_for_quality(extracts)

    # Choose the path: full text or segmented
    if len(report) <= QUALITY_FULLTEXT_THRESHOLD:
        return await _quality_fulltext(
            report, extracts_text, plan_questions_text, llm,
        )
    return await _quality_segmented(
        report, extracts_text, plan_questions_text, llm,
    )


async def _quality_fulltext(
    report: str,
    extracts_text: str,
    plan_questions_text: str,
    llm: "HarvestModelAdapter",
) -> QualityCheckResult:
    prompt = _QUALITY_PROMPT.format(
        plan_questions=plan_questions_text,
        report=report,
        extracts=extracts_text,
    )
    try:
        response = await llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
    except Exception as e:
        logger.warning("check_report_quality failed: %s", e)
        return QualityCheckResult(passed=True, fallback_used=True)

    parsed = parse_llm_json(
        response,
        expected_keys=["passed"],
        default={"passed": True, "issues": []},
    )
    parsed = _english_contract(parsed)
    if not isinstance(parsed, dict):
        return QualityCheckResult(passed=True, fallback_used=True)
    return _build_result(parsed, n_segmente=1)


async def _quality_segmented(
    report: str,
    extracts_text: str,
    plan_questions_text: str,
    llm: "HarvestModelAdapter",
) -> QualityCheckResult:
    """One LLM call per segment, findings are merged.

    All segments receive the SAME plan questions + extracts — otherwise
    segment 2 would not "know" question F1 from segment 1 and would
    wrongly mark it as "missing".
    """
    segments = split_for_quality(report)
    all_maengel: list[dict] = []
    any_failed = False
    ratings: list[str] = []

    for idx, seg in enumerate(segments, start=1):
        prompt = _QUALITY_PROMPT.format(
            plan_questions=plan_questions_text,
            report=(
                f"[Segment {idx}/{len(segments)} of a longer report]\n\n{seg}"
            ),
            extracts=extracts_text,
        )
        try:
            response = await llm.complete(
                [{"role": "user", "content": prompt}], max_tokens=2000,
            )
        except Exception as e:
            logger.warning(
                "Segment quality %d failed: %s", idx, e,
            )
            continue

        parsed = parse_llm_json(
            response,
            expected_keys=["passed"],
            default={"passed": True, "issues": []},
        )
        parsed = _english_contract(parsed)
        if not isinstance(parsed, dict):
            continue

        # Tag the findings with the segment
        for m in parsed.get("issues", []) or []:
            if isinstance(m, dict):
                m_with_seg = dict(m)
                m_with_seg["segment"] = idx
                all_maengel.append(m_with_seg)
        if not parsed.get("passed", True):
            any_failed = True
        if parsed.get("rating"):
            ratings.append(parsed["rating"])

    deduped = dedupe_maengel(all_maengel)

    # Overall rating: with one failed segment enforce "poor",
    # otherwise the most frequent rating
    if any_failed:
        rating = next(
            (b for b in ratings
             if any(k in b.lower() for k in
                    ["poor", "mangel", "kritisch"])),
            (ratings[0] if ratings else "needs_improvement"),
        )
    elif ratings:
        rating = max(ratings, key=ratings.count)
    else:
        rating = ""

    logger.info(
        "Segmented quality check: %d segments, %d consolidated "
        "findings (raw: %d)",
        len(segments), len(deduped), len(all_maengel),
    )

    return QualityCheckResult(
        passed=not any_failed,
        rating=rating,
        issues=deduped,
        segments=len(segments),
    )


# ── Fulfilment check ───────────────────────────────


async def check_query_fulfillment(
    query: str,
    report: str,
    dual_llm,
) -> FulfillmentResult:
    """Check whether the report fulfils the original user request.

    In contrast to the coverage classifier, which works per plan
    question, this function works at the level of the original query:
    does the report as a whole cover what the user wanted to know?

    For long reports an excerpt is taken — the top of the report
    (typically an executive summary or the first sections) plus the end.
    That is enough to answer "was the request answered?" robustly
    without blowing up the prompt.

    Args:
        query: the original user request.
        report: the final report.
        dual_llm: DualLLMClient. Wrapped internally in a
                  HarvestModelAdapter.

    Returns:
        FulfillmentResult. On error or empty inputs: conservatively
        fulfilled=True with fallback_used=True.
    """
    if not query.strip() or not report.strip():
        return FulfillmentResult(fulfilled=True, fallback_used=True)

    from src.llm.classifier_adapter import HarvestModelAdapter
    llm = HarvestModelAdapter(dual_llm)

    # Report excerpt (top + end, if too long)
    if len(report) <= FULFILLMENT_REPORT_MAX:
        excerpt = report
    else:
        head = report[: FULFILLMENT_REPORT_MAX // 2]
        tail = report[-FULFILLMENT_REPORT_MAX // 2:]
        excerpt = head + "\n\n[...]\n\n" + tail

    try:
        response = await llm.complete(
            [{
                "role": "user",
                "content": _FULFILLMENT_PROMPT.format(
                    query=query, report_excerpt=excerpt,
                ),
            }],
            max_tokens=600,
        )
    except Exception as e:
        logger.warning("check_query_fulfillment failed: %s", e)
        return FulfillmentResult(fulfilled=True, fallback_used=True)

    parsed = parse_llm_json(
        response,
        expected_keys=["fulfilled"],
        default={"fulfilled": True, "assessment": "", "rework": ""},
    )
    parsed = _english_contract(parsed)
    return FulfillmentResult.from_dict(parsed)


# ── Helper functions ───────────────────────────────────────────────


def split_for_quality(text: str) -> list[str]:
    """Segment a long report for the quality check.

    Strategy:
      1. if ≥ N Markdown headings exist: cut at headings close to an
         even split.
      2. otherwise: paragraph-aligned (at `\\n\\n`).

    Goal: 2-3 segments, depending on the length.
    """
    if not text:
        return []
    target_segments = 2 if len(text) <= 30_000 else 3

    # Split based on Markdown headings
    heading_positions = [
        m.start() for m in re.finditer(r"^#{1,3}\s", text, re.MULTILINE)
    ]

    if len(heading_positions) >= target_segments:
        ideal_breaks = [
            len(text) * i // target_segments
            for i in range(1, target_segments)
        ]
        actual_breaks = []
        for ideal in ideal_breaks:
            candidates = [p for p in heading_positions if p >= ideal]
            actual_breaks.append(candidates[0] if candidates else ideal)
        actual_breaks = sorted(set(actual_breaks))
        actual_breaks = [0] + actual_breaks + [len(text)]
        segs = [
            text[actual_breaks[i]: actual_breaks[i + 1]]
            for i in range(len(actual_breaks) - 1)
        ]
        return [s for s in segs if s.strip()]

    # Paragraph-based fallback
    paragraphs = text.split("\n\n")
    segment_size = max(1, len(paragraphs) // target_segments)
    return [
        "\n\n".join(paragraphs[i: i + segment_size])
        for i in range(0, len(paragraphs), segment_size)
    ]


def dedupe_maengel(raw_maengel: list) -> list[QualityIssue]:
    """Merge finding lists from several segment checks.

    Findings with the same (normalised) description are merged into one
    QualityIssue; the `segments` field accumulates the segment indices.
    """
    if not raw_maengel:
        return []

    merged: dict[str, QualityIssue] = {}
    plain_strings: list[str] = []

    for m in raw_maengel:
        if not isinstance(m, dict):
            ps = str(m).strip()
            if ps and ps not in plain_strings:
                plain_strings.append(ps)
            continue
        description = str(m.get("description", "")).strip()
        if not description:
            continue
        key = description.lower()[:120]
        seg = m.get("segment")

        if key in merged:
            # Known finding from another segment → accumulate the segment
            if seg is not None and seg not in merged[key].segments:
                merged[key].segments.append(seg)
        else:
            # New finding → via the central conversion
            merged[key] = QualityIssue.from_dict(m)

    result = list(merged.values()) + [
        QualityIssue(description=s) for s in plain_strings
    ]
    return result


def _format_extracts_for_quality(extracts: list) -> str:
    """Format positive extracts for the quality prompt.

    Truncates at EXTRACTS_MAX_FOR_QUALITY so that the prompt does not
    get too large. Negative and meta extracts are NOT used as the
    factual basis (they would confuse the LLM).
    """
    lines = []
    for i, e in enumerate(extracts):
        if getattr(e, "polarity", "positive") != "positive":
            continue
        title = (getattr(e, "source_title", "") or "")[:60]
        fact = (getattr(e, "fact", "") or "")
        qid = getattr(e, "question_id", "")
        lines.append(f"[E{i} | {qid} | {title}] {fact}")
    text = "\n".join(lines)
    if len(text) > EXTRACTS_MAX_FOR_QUALITY:
        # Cut at the end of an entry, not in the middle of a word
        cut = text[: EXTRACTS_MAX_FOR_QUALITY].rfind("\n")
        if cut > 0:
            text = text[: cut] + "\n[...]"
        else:
            text = text[: EXTRACTS_MAX_FOR_QUALITY]
    return text or "(no positive extracts available)"


def _build_result(parsed: dict, n_segmente: int) -> QualityCheckResult:
    """Build a QualityCheckResult from parsed JSON."""
    issues = []
    for m in parsed.get("issues", []) or []:
        if not isinstance(m, dict):
            issues.append(QualityIssue(description=str(m)))
            continue
        issues.append(QualityIssue.from_dict(m))
    return QualityCheckResult(
        passed=bool(parsed.get("passed", True)),
        rating=str(parsed.get("rating", "")),
        issues=issues,
        segments=n_segmente,
    )
