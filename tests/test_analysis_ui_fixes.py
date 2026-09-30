"""Tests for UI behaviour of the decision analysis:
  1. progress lines show node names (not "▶️ Starting: `?` ✅ `?` done")
  2. confirmation gate only when the checkbox is ticked
  3. the report does not start with the complete request as H1
  4. the chat label is not "Decision analysis: decision_analysis"
"""

import tests.conftest  # noqa: F401  (dependency stubs)

from src.pipeline.analysis_pipeline import _report_title
from src.pipeline.dag import NodeResult, NodeStatus
from src.pipeline.models import SubTask, TaskPlan
from src.ui.analysis_runner import (
    format_analysis_event,
    make_chat_summary,
    shorten_for_display,
)
from src.ui.components.plan_preview import should_show_preview

LONG_REQUEST = (
    "Recherchiere ein datenklassengetriebenes KI-Strategiekonzept für ein "
    "großes universitäres Institut mit ca. 1.400 Mitarbeitenden "
    "(200 Professuren, 330 Support-Mitarbeiter, 200 Wissenschaftliche "
    "Mitarbeiter, 700 Doktoranden), das derzeit fragmentiert operiert."
)


# ── 1. Progress events ────────────────────────────────────────

def test_node_start_shows_name_not_questionmark():
    """The DAG sends {"name": ...}; the formatter must accept it, not only {"node": ...}."""
    out = format_analysis_event("node_start", {"name": "layer_1"})
    assert "layer_1" in out and "?" not in out


def test_node_done_accepts_noderesult_object():
    """node_done delivers a NodeResult, not a dict."""
    nr = NodeResult(status=NodeStatus.OK, node_name="layer_1",
                    duration_seconds=2.5,
                    metadata={"done": 4, "executed": 4, "failed": 0})
    out = format_analysis_event("node_done", nr)
    assert "layer_1" in out and "?" not in out
    assert "4/4" in out, "layer metadata should become visible"
    assert "2500 ms" in out


def test_node_failed_shows_error():
    nr = NodeResult(status=NodeStatus.FAILED, node_name="layer_2",
                    error=RuntimeError("Timeout nach 300s"))
    out = format_analysis_event("node_failed", nr)
    assert "layer_2" in out and "Timeout" in out


def test_node_event_without_payload_is_harmless():
    assert format_analysis_event("node_start", None)
    assert format_analysis_event("node_start", {})


def test_explicit_node_key_still_works():
    """Backwards compatible: whoever already sends 'node' is not broken."""
    out = format_analysis_event("node_start", {"node": "eigener_name"})
    assert "eigener_name" in out


# ── 2. Gate only with the checkbox ticked ───────────────────────────

def _decision_plan():
    return TaskPlan(use_case="decision_analysis", tasks=[
        SubTask(id=f"CELL_{i}", phase="cell_evaluation", description="d",
                prompt_template="t") for i in range(4)
    ] + [SubTask(id="REC", phase="recommendation", description="d",
                 prompt_template="t")])


def test_heuristic_alone_would_open_the_gate():
    """Shows why the checkbox is needed: the heuristic applies to every analysis."""
    assert should_show_preview(_decision_plan(), force=False) is True


def test_gate_logic_matches_the_checkbox():
    """Production condition: `show_gate and should_show_preview(plan)`."""
    plan = _decision_plan()
    assert (False and should_show_preview(plan)) is False
    assert (True and should_show_preview(plan)) is True


def test_trivial_plan_stays_closed_even_when_checked():
    """As in the research path: the heuristic filters out trivial plans."""
    trivial = TaskPlan(use_case="x", tasks=[
        SubTask(id="A", phase="p", description="d", prompt_template="t")])
    assert (True and should_show_preview(trivial)) is False


# ── 3. Report title ─────────────────────────────────────────────

def test_title_is_shortened():
    title = _report_title(LONG_REQUEST)
    assert len(title) <= 82
    assert title.endswith("…")
    assert title.startswith("Recherchiere ein datenklassengetriebenes")


def test_title_breaks_at_word_boundary():
    assert not _report_title(LONG_REQUEST).rstrip(" …").endswith("-")
    assert "  " not in _report_title(LONG_REQUEST)


def test_title_uses_only_first_line():
    assert _report_title("Erste Zeile\nZweite Zeile") == "Erste Zeile"


def test_short_title_untouched():
    assert _report_title("Welches CRM-System?") == "Welches CRM-System?"


def test_title_handles_empty():
    assert _report_title("") == "" and _report_title(None) == ""


# ── 4. Chat summary ───────────────────────────────────────

def test_summary_uses_decision_field():
    """decision_analysis must not fall back to the use-case name."""
    out = make_chat_summary("decision_analysis", {
        "decision": LONG_REQUEST, "options": "A\nB", "criteria": "K"})
    assert out != "decision_analysis"
    assert out.startswith("Recherchiere ein")


def test_summary_covers_all_use_case_first_fields():
    cases = {
        "decision_analysis": "decision",
        "research_design": "research_question",
        "grant_proposal": "research_question",
        "peer_review": "manuscript_summary",
        "literature_review": "topic",
        "literature_finder": "topic",
        "explainer": "topic",
    }
    for use_case, field in cases.items():
        out = make_chat_summary(use_case, {field: "Ein sinnvoller Inhalt"})
        assert out == "Ein sinnvoller Inhalt", f"{use_case}/{field}"


def test_summary_falls_back_to_any_text_before_use_case_name():
    out = make_chat_summary("decision_analysis", {"unbekannt": "Irgendein Text"})
    assert out == "Irgendein Text"


def test_summary_last_resort_is_use_case():
    assert make_chat_summary("decision_analysis", {}) == "decision_analysis"


def test_shorten_helper():
    assert shorten_for_display("kurz") == "kurz"
    assert shorten_for_display("a" * 200, 20).endswith("…")
    assert shorten_for_display("") == ""


# ── 5. Plan gate in the research mode ────────────────

def test_research_plan_formatter_accepts_production_arguments():
    """gradio_app calls with search_stats/mode/academic_only.

    The plan gate must accept these keyword arguments (otherwise it fails
    with an unexpected keyword argument as soon as "Confirm plan" is
    ticked).
    """
    from src.institution import InstitutionProfile, set_profile
    from src.pipeline.models import ResearchPlan
    from src.ui.components.plan_preview import format_research_plan_markdown

    set_profile(InstitutionProfile(name="Example University", short_name="EU",
                                   domains=("example.edu",)))
    try:
        out = format_research_plan_markdown(
            ResearchPlan(questions=[]),
            {"langs": ["de", "en"], "n_terms": 12},
            mode="institution",
            academic_only=True,
        )
    finally:
        set_profile(None)
    assert "Rechercheplan" in out
    assert "EU-Recherche" in out
    assert "wissenschaftliche Quellen" in out
    assert "de, en" in out


def test_research_plan_formatter_without_context():
    from src.pipeline.models import ResearchPlan
    from src.ui.components.plan_preview import format_research_plan_markdown

    out = format_research_plan_markdown(ResearchPlan(questions=[]))
    assert "Rechercheplan" in out
    assert "Mode" not in out


def test_research_plan_formatter_handles_none():
    from src.ui.components.plan_preview import format_research_plan_markdown
    assert format_research_plan_markdown(None)
