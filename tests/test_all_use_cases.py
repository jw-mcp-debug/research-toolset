"""
Smoke test of an analysis use case through the complete pipeline:
preflight → decompose → DAG execution → synthesis → footer.

Uses a "smart" mock DualLLMClient that inspects the prompt content and
returns outputs matching the use case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from tests._mocks import MockSubClient, SmartMockLLM  # noqa: F401
from src.pipeline.analysis_pipeline import (
    AnalysisPipelineRunner,
)
from src.pipeline.models import HarvestContext, TaskState


# ─── Smart mock LLM ────────────────────────────────────────────────





# ─── Test helpers ────────────────────────────────────────────────────

async def run_use_case(use_case: str, preflight_data: dict) -> HarvestContext:
    """Run a use case once completely with the mock LLM."""
    mock = SmartMockLLM()
    runner = AnalysisPipelineRunner(
            progress_callback=None,
        llm_client=mock, max_parallel=5, rate_limit=100,
    )
    ctx = HarvestContext(query=preflight_data.get("question") or "test")
    ctx.use_case = use_case
    ctx.preflight_data = preflight_data
    ctx._llm_client = mock
    await runner.run(ctx)
    return ctx


# ─── Tests ──────────────────────────────────────────────────────────

async def test_peer_review_smoke():
    """Peer review runs through with the mock LLM."""
    print("Test: peer_review smoke ...", end=" ")
    paper_text = (
        "Abstract\n\n" + "Einleitung. " * 100 +
        "\n\nMethoden\n\n" + "Methodische Beschreibung. " * 100 +
        "\n\nErgebnisse\n\n" + "Ergebnisbeschreibung. " * 100 +
        "\n\nDiskussion\n\n" + "Diskussion der Befunde. " * 100
    )
    ctx = await run_use_case("peer_review", {
        "paper_text": paper_text,
        "discipline": "Sozialwissenschaften",
        "role": "Journal-Gutachten",
        "focus": "",
    })
    assert ctx.status == "done", f"Status: {ctx.status} ({ctx.error_message})"
    assert ctx.final_report
    assert "Gutachten" in ctx.final_report or "Mock" in ctx.final_report
    done = sum(1 for r in ctx.task_results.values() if r.state == TaskState.DONE)
    print(f"✓ ({done} Tasks done)")
