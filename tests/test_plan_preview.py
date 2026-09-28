"""
Tests for the plan preview helpers (Gradio-independent functions).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.models import (
    TaskPlan, SubTask, ResearchPlan, ResearchQuestion,
)
from src.ui.components.plan_preview import (
    PREVIEW_NEVER,
    extract_editable_items,
    format_plan_markdown,
    format_research_plan_markdown,
    should_show_preview,
    should_show_research_preview,
)


def make_plan(use_case: str, num_tasks: int = 5,
              estimated_calls: int = 10,
              estimated_duration: int = 30) -> TaskPlan:
    """Helper."""
    tasks = [
        SubTask(
            id=f"T{i}",
            phase="test",
            description=f"Task {i}",
            prompt_template="dummy",
        )
        for i in range(num_tasks)
    ]
    return TaskPlan(
        use_case=use_case,
        tasks=tasks,
        estimated_calls=estimated_calls,
        estimated_duration_seconds=estimated_duration,
    )


def test_should_show_preview_auto_over_calls():
    """force=True: the preview is shown (the user ticked the checkbox)."""
    print("Test: should_show_preview force=True (checkbox ticked) ...", end=" ")
    plan = make_plan("explainer", estimated_calls=40)
    assert should_show_preview(plan, force=True) == True
    print("✓")


def test_should_show_preview_always():
    """force=True with a small plan: preview anyway."""
    print("Test: should_show_preview force=True even for a small plan ...", end=" ")
    plan = make_plan("explainer", estimated_calls=2)
    assert should_show_preview(plan, force=True) == True
    print("✓")


def test_should_show_preview_never_normal():
    """NEVER mode: no preview (except with a mandatory gate)."""
    print("Test: should_show_preview NEVER ...", end=" ")
    plan = make_plan("explainer", estimated_calls=200)
    assert should_show_preview(plan, mode=PREVIEW_NEVER) == False
    print("✓")


def test_extract_editable_items_other_use_case():
    """Use case without editable items."""
    print("Test: extract_editable_items empty case ...", end=" ")
    plan = TaskPlan(use_case="literature_review", tasks=[
        SubTask(id="X", phase="screening", description="?", prompt_template="?"),
    ])
    items = extract_editable_items(plan)
    assert items == []
    print("✓")


# ─── Tests for extract_plan_metadata_info (read-only literature search) ──

def test_format_markdown_without_literature_section():
    """A plan without literature metadata shows no literature section."""
    print("Test: format_markdown without literature ...", end=" ")
    plan = TaskPlan(
        use_case="explainer",
        tasks=[
            SubTask(
                id="E", phase="explanation", description="Erkläre X",
                prompt_template="?", prompt_params={"concept": "X"},
            ),
        ],
        estimated_calls=1,
        estimated_duration_seconds=10,
    )
    md = format_plan_markdown(plan)
    assert "Literatursuche" not in md
    print("✓")


# ─── Tests for format_research_plan_markdown + should_show_research_preview ──

def test_format_research_plan_institution_mode():
    """The institution research mode is shown in the header."""
    print("Test: format_research_plan institution mode ...", end=" ")
    plan = ResearchPlan(
        summary="Institutions-Recherche zur Person",
        questions=[
            ResearchQuestion(
                id="F1",
                question="Wer ist X?",
                search_langs=["de"],
                search_terms_by_lang={"de": ["X site:example.edu"]},
            ),
        ],
    )
    md = format_research_plan_markdown(plan, None, mode="institution")
    assert "Institutions-Recherche" in md
    assert "Wer ist X?" in md
    print("✓")


def test_should_show_research_preview_empty_plan():
    """A plan without questions (or None) shows no preview."""
    print("Test: should_show_research_preview empty ...", end=" ")
    assert should_show_research_preview(None) is False
    empty = ResearchPlan(summary="", questions=[])
    assert should_show_research_preview(empty) is False
    print("✓")


def main():
    print("=" * 60)
    print("Plan-Preview-Helper Tests")
    print("=" * 60)
    tests = [
        test_should_show_preview_auto_over_calls,
        test_should_show_preview_always,
        test_should_show_preview_never_normal,
        test_extract_editable_items_other_use_case,
        # extract_plan_metadata_info + literature section
        test_format_markdown_without_literature_section,
        # research plan rendering + gate decision
        test_format_research_plan_institution_mode,
        test_should_show_research_preview_empty_plan,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {test.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print("=" * 60)
    if failed:
        print(f"❌ {failed}/{len(tests)} tests failed")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
