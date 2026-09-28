"""
Tests for the plan-gate parameters of the ResearchOrchestrator.

Checks the integration between the orchestrator and the
AnalysisPipelineRunner:
- plan_only=True returns a context with task_plan, without execution
- existing_ctx + skip_decompose=True uses the previous plan
- the two-phase flow (plan_only → skip_decompose) gives the same final
  result as a single-phase call
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Dependency mocks — the orchestrator imports some modules that are not
# available in the test sandbox. We mock them the same way as
# test_all_use_cases does (which only tests the runner directly).
if "httpx" not in sys.modules:
    sys.modules["httpx"] = MagicMock()
if "src.connectors.rate_limiter" not in sys.modules:
    _rl = MagicMock()
    _rl.get_rate_limiter = MagicMock(return_value=MagicMock())
    sys.modules["src.connectors.rate_limiter"] = _rl
# src.config: AppConfig and PipelineConfig are imported, but not used
# for the analysis path
if "src.config" not in sys.modules:
    _cfg = MagicMock()

    class _AppConfig:
        pass

    class _PipelineConfig:
        max_rounds = 3
        max_parallel_fetches = 5
        max_content_chars = 50000
        min_content_chars = 100
        fetch_timeout_seconds = 30
        dedupe_threshold = 0.85

    _cfg.AppConfig = _AppConfig
    _cfg.PipelineConfig = _PipelineConfig
    sys.modules["src.config"] = _cfg
# src.connectors.base — imports normalize_url, is_url_blocked etc.
if "src.connectors.base" not in sys.modules:
    _base = MagicMock()
    _base.ConnectorRegistry = MagicMock
    _base.normalize_url = lambda u: u
    _base.is_url_blocked = lambda u: False
    sys.modules["src.connectors.base"] = _base
# src.llm.client
if "src.llm.client" not in sys.modules:
    _llm_mod = MagicMock()
    _llm_mod.DualLLMClient = MagicMock
    sys.modules["src.llm.client"] = _llm_mod

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.orchestrator import ResearchOrchestrator  # noqa: E402
from src.pipeline.models import TaskState  # noqa: E402

# SmartMockLLM from the shared test doubles
from tests._mocks import SmartMockLLM  # noqa: E402


# ─── Minimal-Config & Connector-Mocks ─────────────────────────────


class MockConnectors:
    """Minimal connector stub — enough for the analysis mode, because the
    analysis pipeline does no web research."""
    def list(self):
        return []


class MockPipelineConfig:
    max_rounds = 3
    max_parallel_fetches = 5
    max_content_chars = 50000
    min_content_chars = 100
    fetch_timeout_seconds = 30
    dedupe_threshold = 0.85


def _make_orchestrator():
    llm = SmartMockLLM()
    return ResearchOrchestrator(
        llm=llm,
        connectors=MockConnectors(),
        config=MockPipelineConfig(),
        person_directory=None,
    )


def _peer_review_inputs():
    paper_text = (
        "Abstract\n\n" + "Einleitung zur Studie und Hintergrund. " * 50 +
        "\n\nMethoden\n\n" + "Methodische Beschreibung der Studie. " * 50 +
        "\n\nErgebnisse\n\n" + "Ergebnisbeschreibung und Analyse. " * 50 +
        "\n\nDiskussion\n\n" + "Diskussion der Befunde im Kontext. " * 50
    )
    return {
        "paper_text": paper_text,
        "discipline": "Sozialwissenschaften",
        "role": "Journal-Gutachten",
        "focus": "",
    }


async def _silent_progress(event, data):
    pass


# ─── Tests ────────────────────────────────────────────────────────


async def test_plan_only_returns_ctx_with_plan():
    """orchestrator.run(plan_only=True) returns a context with task_plan."""
    print("Test: plan_only returns a plan without execution ...", end=" ")
    orch = _make_orchestrator()
    ctx = await orch.run(
        query="test",
        chat_history=[],
        context_docs="",
        template_name="Peer Review",
        progress_callback=_silent_progress,
        mode="peer_review",
        preflight_data=_peer_review_inputs(),
        plan_only=True,
    )
    assert ctx is not None
    assert ctx.task_plan is not None, (
        f"task_plan missing: status={ctx.status}, error={ctx.error_message}"
    )
    assert len(ctx.task_plan.tasks) > 0
    # the pipeline was NOT run
    assert ctx.status != "done", f"status should not be done: {ctx.status}"
    assert not ctx.task_results or len(ctx.task_results) == 0
    # finished_at must not be set yet
    assert not getattr(ctx, "finished_at", None)
    print(f"✓ ({len(ctx.task_plan.tasks)} Tasks)")


async def test_plan_only_then_resume():
    """Two-phase flow: plan_only → existing_ctx + skip_decompose."""
    print("Test: plan_only → resume returns the final result ...", end=" ")
    orch = _make_orchestrator()

    # Phase 1: build the plan
    ctx = await orch.run(
        query="test",
        chat_history=[],
        context_docs="",
        template_name="Peer Review",
        progress_callback=_silent_progress,
        mode="peer_review",
        preflight_data=_peer_review_inputs(),
        plan_only=True,
    )
    assert ctx.task_plan is not None
    plan_task_count = len(ctx.task_plan.tasks)

    # Phase 2: run the plan (same orchestrator, same ctx)
    ctx2 = await orch.run(
        query="test",
        chat_history=[],
        context_docs="",
        template_name="Peer Review",
        progress_callback=_silent_progress,
        mode="peer_review",
        existing_ctx=ctx,
        skip_decompose=True,
    )
    # the same context must come back
    assert ctx2 is ctx
    # the pipeline is now "done"
    assert ctx.status == "done", (
        f"Status: {ctx.status}, error: {ctx.error_message}"
    )
    assert ctx.final_report
    assert len(ctx.task_results) > 0
    # the task plan is the same
    assert len(ctx.task_plan.tasks) == plan_task_count
    assert ctx.finished_at
    print(f"✓ ({len(ctx.task_results)} Tasks done)")


async def test_normal_run_unchanged():
    """A normal run() without the new parameters works unchanged."""
    print("Test: normal orchestrator.run() unchanged ...", end=" ")
    orch = _make_orchestrator()
    ctx = await orch.run(
        query="test",
        chat_history=[],
        context_docs="",
        template_name="Peer Review",
        progress_callback=_silent_progress,
        mode="peer_review",
        preflight_data=_peer_review_inputs(),
    )
    assert ctx.status == "done", f"{ctx.status}: {ctx.error_message}"
    assert ctx.final_report
    assert ctx.finished_at
    print("✓")


async def test_two_phase_same_as_single_phase():
    """The two-phase flow gives the same final result as the single-phase one."""
    print("Test: two vs. one phase, same result ...", end=" ")

    # single-phase
    orch_a = _make_orchestrator()
    ctx_a = await orch_a.run(
        query="test",
        chat_history=[],
        context_docs="",
        template_name="Peer Review",
        progress_callback=_silent_progress,
        mode="peer_review",
        preflight_data=_peer_review_inputs(),
    )

    # two-phase
    orch_b = _make_orchestrator()
    ctx_b = await orch_b.run(
        query="test",
        chat_history=[],
        context_docs="",
        template_name="Peer Review",
        progress_callback=_silent_progress,
        mode="peer_review",
        preflight_data=_peer_review_inputs(),
        plan_only=True,
    )
    ctx_b = await orch_b.run(
        query="test",
        chat_history=[],
        context_docs="",
        template_name="Peer Review",
        progress_callback=_silent_progress,
        mode="peer_review",
        existing_ctx=ctx_b,
        skip_decompose=True,
    )

    assert ctx_a.status == "done"
    assert ctx_b.status == "done"
    assert len(ctx_a.task_plan.tasks) == len(ctx_b.task_plan.tasks)
    done_a = sum(
        1 for r in ctx_a.task_results.values()
        if r.state == TaskState.DONE
    )
    done_b = sum(
        1 for r in ctx_b.task_results.values()
        if r.state == TaskState.DONE
    )
    assert done_a == done_b, f"single-phase {done_a} vs two-phase {done_b}"
    print(f"✓ (both {done_a} done)")


async def main():
    print("=" * 60)
    print("Orchestrator Plan-Gate Tests")
    print("=" * 60)

    tests = [
        test_plan_only_returns_ctx_with_plan,
        test_plan_only_then_resume,
        test_normal_run_unchanged,
        test_two_phase_same_as_single_phase,
    ]

    failed = 0
    for t in tests:
        try:
            await t()
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    if failed:
        print(f"❌ {failed}/{len(tests)} tests failed")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
