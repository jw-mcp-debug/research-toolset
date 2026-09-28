"""
Tests for the report footer generator (production details in the output
language of the run).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import output_language
from src.exporters.report_footer import (
    USE_CASE_LABELS,
    generate_markdown_footer,
)
from src.pipeline.analysis_pipeline import USE_CASE_REGISTRY
from src.pipeline.models import (
    HarvestContext,
    SubTask,
    TaskPlan,
    TaskResult,
    TaskState,
)


@pytest.fixture
def german(monkeypatch):
    monkeypatch.setenv("OUTPUT_LANGUAGES", "en,de")
    output_language.reload()
    yield
    monkeypatch.delenv("OUTPUT_LANGUAGES")
    output_language.reload()


def test_footer_minimal():
    """Footer with minimal data."""
    ctx = HarvestContext(query="test")
    footer = generate_markdown_footer(ctx)
    assert "How this report was produced" in footer
    from src.about import TOOL_NAME
    assert TOOL_NAME in footer
    assert "Important:" in footer


def test_footer_with_llm_usage():
    """Footer with LLM statistics."""
    ctx = HarvestContext(query="test")
    ctx.llm_usage = {
        "primary": {"model": "Qwen3-235B", "requests": 5,
                    "prompt_tokens": 10000, "completion_tokens": 2000},
        "harvest": {"model": "Qwen3-30B", "requests": 12,
                    "prompt_tokens": 30000, "completion_tokens": 5000},
    }
    footer = generate_markdown_footer(ctx)
    assert "17" in footer  # 5 + 12 calls
    assert "Qwen3-235B" in footer
    assert "Qwen3-30B" in footer
    assert "40,000" in footer  # input tokens
    assert "7,000" in footer   # output tokens


def test_footer_explainer_use_case():
    """Footer with explainer-specific data from the active task plan."""
    ctx = HarvestContext(query="Explain LLMs")
    ctx.use_case = "explainer"
    ctx.preflight_data = {
        "topic": "How do LLMs work?",
        "audience": "Undergraduates",
    }
    ctx.task_plan = TaskPlan(use_case="explainer", tasks=[
        SubTask(id="EXP_1", phase="explanation", description="?",
                prompt_template="?", prompt_params={"concept": "Tokenisation"}),
        SubTask(id="EXP_2", phase="explanation", description="?",
                prompt_template="?", prompt_params={"concept": "Embeddings"}),
    ])
    ctx.task_results = {
        "EXP_1": TaskResult(task_id="EXP_1", state=TaskState.DONE, tokens_used=300),
        "EXP_2": TaskResult(task_id="EXP_2", state=TaskState.DONE, tokens_used=350),
    }
    footer = generate_markdown_footer(ctx)
    assert "In-depth explanation" in footer
    assert "How do LLMs work?" in footer
    assert "Undergraduates" in footer
    assert "Tokenisation" in footer and "Embeddings" in footer
    assert "2 successful" in footer


def test_footer_with_failed_tasks_and_counts_lines():
    """Failed/skipped tasks are counted; options are counted per line, not per character."""
    ctx = HarvestContext(query="test")
    ctx.use_case = "decision_analysis"
    ctx.task_results = {
        "T1": TaskResult(task_id="T1", state=TaskState.DONE),
        "T2": TaskResult(task_id="T2", state=TaskState.FAILED, error="?"),
        "T3": TaskResult(task_id="T3", state=TaskState.SKIPPED),
    }
    ctx.preflight_data = {"options": "Option A\nOption B", "criteria": "c1\nc2\nc3"}
    footer = generate_markdown_footer(ctx)
    assert "Decision analysis" in footer
    assert "1 successful" in footer and "1 failed" in footer and "1 skipped" in footer
    assert "**Number of options:** 2" in footer
    assert "**Number of criteria:** 3" in footer


def test_footer_grant_proposal():
    """Footer for a grant proposal shows the funder and duration."""
    ctx = HarvestContext(query="test")
    ctx.use_case = "grant_proposal"
    ctx.preflight_data = {"funder": "Example Foundation", "duration": "36 months"}
    footer = generate_markdown_footer(ctx)
    assert "Grant proposal" in footer
    assert "Example Foundation" in footer
    assert "36 months" in footer


def test_plan_confirmation_only_when_confirmed():
    """'Confirmed by the user' appears only if the plan gate was actually passed."""
    ctx = HarvestContext(query="test")
    ctx.use_case = "explainer"
    ctx.task_plan = TaskPlan(use_case="explainer", tasks=[])
    assert "confirmed by the user" not in generate_markdown_footer(ctx)
    ctx.plan_confirmed = True
    assert "confirmed by the user" in generate_markdown_footer(ctx)


def test_footer_in_german(german):
    """With German as the output language the footer is German."""
    ctx = HarvestContext(query="test")
    ctx.use_case = "decision_analysis"
    ctx.output_language = "de"
    ctx.preflight_data = {"options": "A\nB"}
    footer = generate_markdown_footer(ctx)
    assert "Hinweise zur Erstellung" in footer
    assert "Entscheidungsanalyse" in footer
    assert "Anzahl Optionen" in footer
    assert "Wichtiger Hinweis" in footer


def test_use_case_labels_complete():
    """Every analysis use case has a label."""
    for uc in USE_CASE_REGISTRY:
        assert uc in USE_CASE_LABELS, f"label missing: {uc}"
