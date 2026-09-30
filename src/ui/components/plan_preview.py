"""
Plan preview components for analysis and research plans.

Used by `gradio_app.py` to render plans in the chat history and to
decide whether a plan gate should be shown to the user at all (very
trivial plans can skip the gate).

The functions are pure string/data transformations, fully unit-testable
and Gradio-agnostic. A UI component combines them with a `gr.Markdown`
output.

There are two plan types:

  - **TaskPlan** (analysis pipeline) — phases with sub-tasks and
    dependencies. `format_task_plan_markdown()`, `should_show_preview()`,
    `extract_plan_metadata_info()`.

  - **ResearchPlan** (research pipeline) — questions, direct_urls,
    git_repos. `format_research_plan_markdown()` (delegates to
    `src.ui.plan_editor`), `should_show_research_preview()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.pipeline.models import ResearchPlan, TaskPlan


# ── Preview modes (deprecated, but importable) ──
#
# AUTO (decide heuristically), ALWAYS (always show), NEVER (never
# show). The simple `force: bool` argument of `should_show_preview`
# covers this; the constants only exist so that test modules that
# import them keep working — the production path does not use them.
PREVIEW_AUTO = "auto"
PREVIEW_ALWAYS = "always"
PREVIEW_NEVER = "never"


# ── TaskPlan (analysis) ───────────────────────────────────────────


# Alias for `format_task_plan_markdown` — some test modules import
# `format_plan_markdown`. Kept so that these imports keep working;
# the production path uses the explicit name.
def format_plan_markdown(plan: Optional["TaskPlan"]) -> str:
    """Alias for format_task_plan_markdown."""
    return format_task_plan_markdown(plan)


def extract_editable_items(plan: Optional["TaskPlan"]) -> list:
    """Deprecated no-op: per-task editing of the plan is not offered.

    Returns an empty list so that test imports keep working.
    """
    return []


def apply_user_edits(plan: Optional["TaskPlan"], edits: dict) -> "TaskPlan":
    """Deprecated no-op counterpart of extract_editable_items.
    Returns the plan unchanged.
    """
    return plan


def format_task_plan_markdown(plan: Optional["TaskPlan"]) -> str:
    """TaskPlan as a Markdown string.

    Returns a readable preview for the chat history:
      - header with use case + number of tasks
      - grouping by phase
      - dependency notes

    Counterpart: `format_research_plan_markdown` for `ResearchPlan`
    (classic research plans). The explicitly separate names prevent the
    wrong type from being rendered after an import.
    """
    if plan is None:
        return "_(kein Analyseplan)_"
    if not plan.tasks:
        return f"### Plan: {plan.use_case}\n\n_(noch keine Aufgaben)_"

    # Group tasks by phase
    by_phase: dict[str, list] = {}
    phase_order: list[str] = []
    for t in plan.tasks:
        if t.phase not in by_phase:
            by_phase[t.phase] = []
            phase_order.append(t.phase)
        by_phase[t.phase].append(t)

    lines: list[str] = []
    lines.append(f"### 📋 Analyseplan: {plan.use_case}")
    lines.append("")
    lines.append(f"_{len(plan.tasks)} Aufgaben in {len(phase_order)} Phasen_")
    lines.append("")

    for phase_idx, phase in enumerate(phase_order, start=1):
        tasks = by_phase[phase]
        lines.append(f"**Phase {phase_idx}: {phase}** ({len(tasks)} {'Aufgabe' if len(tasks) == 1 else 'Aufgaben'})")
        for t in tasks:
            desc = t.description or t.id
            deps = ""
            if t.depends_on:
                deps = f" _(← {', '.join(t.depends_on)})_"
            lines.append(f"- `{t.id}` — {desc}{deps}")
        lines.append("")

    if plan.estimated_calls:
        lines.append(f"_Geschätzt: {plan.estimated_calls} LLM-Aufrufe_")

    return "\n".join(lines).rstrip()


def should_show_preview(
    plan: Optional["TaskPlan"],
    force: bool = False,
    mode: Optional[str] = None,
) -> bool:
    """Should the plan preview gate be shown to the user?

    Heuristic:
      - `force=True` → always yes (the user explicitly enabled the plan preview).
      - plan is None or has no tasks → no (nothing to show).
      - plan has ≥ 3 tasks OR ≥ 2 phases → yes (not trivial).
      - otherwise → no (a trivial plan would only get in the way).

    The optional `mode` parameter (`PREVIEW_AUTO`/`ALWAYS`/`NEVER`) only
    exists for tests that still pass it:
      - PREVIEW_ALWAYS → like force=True
      - PREVIEW_NEVER  → always False
      - PREVIEW_AUTO or None → default heuristic
    The production path uses `force`.
    """
    # mode takes precedence if set explicitly
    if mode == PREVIEW_ALWAYS:
        return True
    if mode == PREVIEW_NEVER:
        return False
    # mode == PREVIEW_AUTO or None → default heuristic

    if force:
        return True
    if plan is None or not plan.tasks:
        return False
    n_tasks = len(plan.tasks)
    n_phases = len({t.phase for t in plan.tasks})
    return n_tasks >= 3 or n_phases >= 2


def extract_plan_metadata_info(
    plan: Optional["TaskPlan"],
) -> dict:
    """Return plan metadata for the header display.

    Returns:
        Dict with the keys `n_tasks`, `n_phases`, `phases` (list),
        `estimated_calls`, `estimated_duration_seconds`, `use_case`.
        An empty dict if `plan is None`.
    """
    if plan is None:
        return {}

    phases = []
    seen: set[str] = set()
    for t in plan.tasks:
        if t.phase not in seen:
            phases.append(t.phase)
            seen.add(t.phase)

    return {
        "use_case": plan.use_case,
        "n_tasks": len(plan.tasks),
        "n_phases": len(phases),
        "phases": phases,
        "estimated_calls": plan.estimated_calls,
        "estimated_duration_seconds": plan.estimated_duration_seconds,
    }


# ── ResearchPlan (research) ─────────────────────────────────────


def format_research_plan_markdown(
    plan: Optional["ResearchPlan"],
    search_stats: Optional[dict] = None,
    mode: Optional[str] = None,
    academic_only: bool = False,
) -> str:
    """ResearchPlan as a Markdown string.

    Delegates the plan body to `src.ui.plan_editor.format_plan_markdown`
    and adds the run context (mode, filter, search statistics).

    `gradio_app` calls this function in two places with `search_stats`,
    `mode` and `academic_only`; they are accepted and shown as context
    for the confirmation (the plan gate would otherwise fail with an
    unexpected keyword argument as soon as the user asks to confirm the
    plan).
    """
    from src.ui.plan_editor import format_plan_markdown as _fmt
    body = _fmt(plan)

    context: list[str] = []
    if mode:
        from src.institution import get_profile
        label = {"institution": f"{get_profile().label}-Recherche",
                 "web": "Webrecherche"}.get(mode, mode)
        context.append(f"**Modus:** {label}")
    if academic_only:
        context.append("**Filter:** nur wissenschaftliche Quellen")
    if search_stats:
        langs = search_stats.get("langs") or search_stats.get("languages")
        if langs:
            langs_text = ", ".join(str(x) for x in langs) if isinstance(
                langs, (list, tuple, set)) else str(langs)
            context.append(f"**Sprachen:** {langs_text}")
        n_terms = search_stats.get("n_terms") or search_stats.get("terms")
        if isinstance(n_terms, int):
            context.append(f"**Suchbegriffe:** {n_terms}")

    if not context:
        return body
    return body + "\n\n" + " · ".join(context)


def should_show_research_preview(
    plan: Optional["ResearchPlan"],
) -> bool:
    """Should the plan preview gate be shown for a research run?

    Heuristic:
      - plan is None or has neither questions nor direct_urls/git_repos
        → no.
      - plan has ≥ 2 questions OR ≥ 1 question + additional sources
        (direct_urls/git_repos/directory_queries) → yes.
      - plan has exactly 1 question and nothing else → no (trivial).
    """
    if plan is None:
        return False
    n_questions = len(plan.questions)
    n_extras = (
        len(plan.direct_urls)
        + len(plan.git_repos)
        + len(plan.directory_queries)
    )
    if n_questions == 0 and n_extras == 0:
        return False
    if n_questions >= 2:
        return True
    if n_questions >= 1 and n_extras >= 1:
        return True
    return False
