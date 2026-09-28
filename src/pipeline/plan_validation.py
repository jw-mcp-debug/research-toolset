"""
Validation of research plans.

Checks the plan against the schema of `ResearchPlan` and the conventions
the analysis prompt asks the LLM to follow.

Purpose:
  Detect early when the analysis LLM delivers a plan that does not
  follow the conventions. Examples:
    - a plan without questions (the LLM misunderstood the task)
    - a question without search_terms (the research cannot start)
    - an invalid priority value ("very_high" instead of "high")
    - an empty summary (the report has no context)
    - an unknown source_scope

Validation result:
  A list of structured issue objects with category, severity (`error`,
  `warn`), location (which question is affected?) and message. The caller
  decides what to do with it — abort or attempt a repair on ERROR issues,
  only log WARN issues.

This function is deliberately LLM-FREE: a fast, deterministic schema
check. Judging the plan's content is the job of the coverage classifier,
which runs after the harvest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from src.core.vocab import LEVELS, normalize_level

if TYPE_CHECKING:
    from src.pipeline.models import ResearchPlan, ResearchQuestion


# ── Conventions — what a conforming plan should have ─────────────


VALID_PRIORITIES = frozenset(LEVELS)
VALID_SOURCE_SCOPES = frozenset({
    "web", "github", "gitlab", "local", "elastic", "academic", "directory",
})
VALID_LANG_CODES = frozenset({
    "de", "en", "fr", "es", "it", "pt", "nl",
    "pl", "ru", "uk", "tr", "ar", "zh", "ja", "ko",
})

# Thresholds — exceeding them yields a WARN
MIN_QUESTIONS_RECOMMENDED = 1
MAX_QUESTIONS_RECOMMENDED = 12   # more is usually over-engineering
MIN_SEARCH_TERMS_PER_QUESTION = 1
MAX_SEARCH_TERMS_PER_QUESTION = 8   # too many fragment the search
MAX_SUMMARY_LENGTH = 1000
MIN_SUMMARY_LENGTH = 10


# ── Issue data type ────────────────────────────────────────────────


@dataclass
class PlanValidationIssue:
    """One observation of the plan validation.

    Fields:
      severity: "error" if the plan cannot run like this,
                "warn" if it is suboptimal but works in principle.
      category: machine-readable key ("missing_field",
                "invalid_value", "out_of_range", "schema_violation").
      message:  human-readable description.
      location: path to the place ("plan.summary", "plan.questions[2].priority").
    """
    severity: str           # "error" | "warn"
    category: str
    message: str
    location: str = ""

    def is_error(self) -> bool:
        return self.severity == "error"

    def is_warning(self) -> bool:
        return self.severity == "warn"


# ── Main function ─────────────────────────────────────────────────


def validate_plan_conventions(plan: "ResearchPlan") -> list[PlanValidationIssue]:
    """Check a research plan against the structural conventions.

    The check is deterministic and LLM-free. It typically takes
    milliseconds and can be called right after `_run_analysis`, before
    the expensive research loop starts.

    Args:
        plan: the plan to check.

    Returns:
        List of issues. Empty = the plan conforms.
        The caller can filter by `severity == 'error'` to detect abort
        conditions.
    """
    issues: list[PlanValidationIssue] = []

    # ── Plan level ──
    issues.extend(_validate_summary(plan))
    issues.extend(_validate_questions_count(plan))
    issues.extend(_validate_direct_urls(plan))

    # ── Per question ──
    for idx, q in enumerate(plan.questions):
        issues.extend(_validate_question(q, idx))

    # ── Plan-wide ──
    issues.extend(_validate_question_id_uniqueness(plan))

    return issues


# ── Plan level ────────────────────────────────────────────────────


def _validate_summary(plan: "ResearchPlan") -> list[PlanValidationIssue]:
    issues = []
    summary = (plan.summary or "").strip()
    if not summary:
        issues.append(PlanValidationIssue(
            severity="warn",
            category="missing_field",
            message="The plan has no summary — the report will be synthesised without "
                    "an introduction, which reduces readability.",
            location="plan.summary",
        ))
    elif len(summary) < MIN_SUMMARY_LENGTH:
        issues.append(PlanValidationIssue(
            severity="warn",
            category="out_of_range",
            message=f"The summary has only {len(summary)} characters — "
                    f"at least {MIN_SUMMARY_LENGTH} are recommended.",
            location="plan.summary",
        ))
    elif len(summary) > MAX_SUMMARY_LENGTH:
        issues.append(PlanValidationIssue(
            severity="warn",
            category="out_of_range",
            message=f"The summary has {len(summary)} characters — "
                    f"a limit of {MAX_SUMMARY_LENGTH} is recommended.",
            location="plan.summary",
        ))
    return issues


def _validate_questions_count(plan: "ResearchPlan") -> list[PlanValidationIssue]:
    n = len(plan.questions)
    if n < MIN_QUESTIONS_RECOMMENDED:
        # Threshold question: without questions AND without direct_urls/git_repos
        # nothing can be searched at all — that is an error.
        has_alternative = (
            len(plan.direct_urls) > 0
            or len(plan.git_repos) > 0
            or len(plan.directory_queries) > 0
        )
        return [PlanValidationIssue(
            severity="error" if not has_alternative else "warn",
            category="schema_violation",
            message=f"The plan has {n} questions — at least {MIN_QUESTIONS_RECOMMENDED} "
                    f"expected."
                    + ("" if has_alternative
                       else " No direct_urls/git_repos/directory_queries either —"
                            " the research has nothing to do."),
            location="plan.questions",
        )]
    if n > MAX_QUESTIONS_RECOMMENDED:
        return [PlanValidationIssue(
            severity="warn",
            category="out_of_range",
            message=f"The plan has {n} questions — more than {MAX_QUESTIONS_RECOMMENDED} "
                    f"leads to over-engineering and long research runs.",
            location="plan.questions",
        )]
    return []


def _validate_direct_urls(plan: "ResearchPlan") -> list[PlanValidationIssue]:
    from src.core.url_security import is_safe_public_url

    issues = []
    for idx, du in enumerate(plan.direct_urls):
        url = (du.url or "").strip()
        if not url:
            issues.append(PlanValidationIssue(
                severity="error",
                category="missing_field",
                message="direct URL without a URL field",
                location=f"plan.direct_urls[{idx}].url",
            ))
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            issues.append(PlanValidationIssue(
                severity="error",
                category="invalid_value",
                message=f"direct URL has no http(s) scheme: {url[:80]!r}",
                location=f"plan.direct_urls[{idx}].url",
            ))
            continue
        # SSRF protection: no internal URLs
        ok, reason = is_safe_public_url(url)
        if not ok:
            issues.append(PlanValidationIssue(
                severity="error",
                category="invalid_value",
                message=f"direct URL not allowed ({reason}): {url[:80]!r}",
                location=f"plan.direct_urls[{idx}].url",
            ))
    return issues


# ── Question level ───────────────────────────────────────────────────


def _validate_question(
    q: "ResearchQuestion", idx: int,
) -> list[PlanValidationIssue]:
    issues = []
    loc_prefix = f"plan.questions[{idx}]"

    # ID
    qid = (q.id or "").strip()
    if not qid:
        issues.append(PlanValidationIssue(
            severity="error",
            category="missing_field",
            message="question without an ID — extracts cannot be assigned",
            location=f"{loc_prefix}.id",
        ))

    # Question-Text
    text = (q.question or "").strip()
    if not text:
        issues.append(PlanValidationIssue(
            severity="error",
            category="missing_field",
            message="question without text",
            location=f"{loc_prefix}.question",
        ))

    # Search-Terms
    terms = q.search_terms or []
    if not terms and not q.search_terms_by_lang:
        issues.append(PlanValidationIssue(
            severity="error",
            category="missing_field",
            message="the question has neither search_terms nor search_terms_by_lang — "
                    "the search cannot start",
            location=f"{loc_prefix}.search_terms",
        ))
    elif len(terms) > MAX_SEARCH_TERMS_PER_QUESTION:
        issues.append(PlanValidationIssue(
            severity="warn",
            category="out_of_range",
            message=f"the question has {len(terms)} search_terms — "
                    f"at most {MAX_SEARCH_TERMS_PER_QUESTION} are recommended.",
            location=f"{loc_prefix}.search_terms",
        ))

    # Priority
    if q.priority and normalize_level(q.priority, default=None) is None:
        issues.append(PlanValidationIssue(
            severity="error",
            category="invalid_value",
            message=f"invalid priority {q.priority!r}, "
                    f"allowed: {sorted(VALID_PRIORITIES)}",
            location=f"{loc_prefix}.priority",
        ))

    # Source-Scope
    if q.source_scope and q.source_scope not in VALID_SOURCE_SCOPES:
        issues.append(PlanValidationIssue(
            severity="warn",
            category="invalid_value",
            message=f"unknown source_scope {q.source_scope!r}, "
                    f"allowed: {sorted(VALID_SOURCE_SCOPES)}",
            location=f"{loc_prefix}.source_scope",
        ))

    # Search languages (only if set explicitly — some LLMs omit them)
    if q.search_langs:
        for lang in q.search_langs:
            if not isinstance(lang, str):
                issues.append(PlanValidationIssue(
                    severity="error",
                    category="invalid_value",
                    message=f"search_langs entry is not a string: {lang!r}",
                    location=f"{loc_prefix}.search_langs",
                ))
                continue
            if lang.lower() not in VALID_LANG_CODES:
                # Only a warning here — the LLM may have reasons for exotic
                # languages; we do not stop the run
                issues.append(PlanValidationIssue(
                    severity="warn",
                    category="invalid_value",
                    message=f"unknown language code {lang!r}",
                    location=f"{loc_prefix}.search_langs",
                ))

    return issues


# ── Plan-wide checks ──────────────────────────────────


def _validate_question_id_uniqueness(
    plan: "ResearchPlan",
) -> list[PlanValidationIssue]:
    """Question IDs must be unique, otherwise extracts cannot be
    assigned cleanly."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for q in plan.questions:
        qid = (q.id or "").strip()
        if qid:
            if qid in seen:
                duplicates.add(qid)
            seen.add(qid)
    if duplicates:
        return [PlanValidationIssue(
            severity="error",
            category="schema_violation",
            message=f"duplicate question IDs found: {sorted(duplicates)} — "
                    f"extracts cannot be assigned unambiguously.",
            location="plan.questions",
        )]
    return []


# ── Convenience ──────────────────────────────────────────────────


def has_blocking_errors(issues: list[PlanValidationIssue]) -> bool:
    """True if at least one issue has severity='error'."""
    return any(i.is_error() for i in issues)


def format_issues_for_log(issues: list[PlanValidationIssue]) -> str:
    """Format issues for log output (one line per issue)."""
    if not issues:
        return "The plan conforms — no issues."
    lines = []
    for i in issues:
        marker = "❌" if i.is_error() else "⚠️ "
        loc = f" [{i.location}]" if i.location else ""
        lines.append(f"{marker} {i.message}{loc}")
    return "\n".join(lines)
