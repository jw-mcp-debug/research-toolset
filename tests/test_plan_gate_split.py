"""
Tests for splitting the runner for the plan gate:
- decompose_only() returns the plan without executing it
- run(skip_decompose=True) skips the decomposer phase
- a two-phase run (decompose_only → run(skip_decompose=True)) gives the
  same final result as a normal run().

This interruption is the basis of the plan preview gate: the UI can stop
between decompose and execute, show the plan and let the user confirm it.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Dependency mocks as in test_all_use_cases
if "httpx" not in sys.modules:
    sys.modules["httpx"] = MagicMock()
if "src.connectors.rate_limiter" not in sys.modules:
    _rl = MagicMock()
    _rl.get_rate_limiter = MagicMock(return_value=MagicMock())
    sys.modules["src.connectors.rate_limiter"] = _rl

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


from src.pipeline.analysis_pipeline import (  # noqa: E402
    AnalysisPipelineRunner,
)
from src.pipeline.models import HarvestContext, TaskState  # noqa: E402

# Reuse the shared SmartMockLLM — it covers all analysis use cases
# (peer review, decision, explainer, ...)
from tests._mocks import SmartMockLLM  # noqa: E402


# ─── Mock-LLM ──────────────────────────────────────────────────────


class MockSubClient:
    def __init__(self, name: str):
        self.model_name = name
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0


class SimpleMockLLM:
    """Minimal mock for the peer-review pipeline."""

    def __init__(self):
        self.primary = MockSubClient("mock-primary")
        self.harvest = MockSubClient("mock-harvest")

    def _acc(self, client):
        client.total_requests += 1
        client.total_prompt_tokens += 100
        client.total_completion_tokens += 50

    async def primary_complete(self, messages, max_tokens=None,
                                enable_thinking=None) -> str:
        self._acc(self.primary)
        return self._respond(messages[0]["content"])

    async def primary_complete_json(self, messages, max_tokens=None) -> dict:
        self._acc(self.primary)
        return self._respond_json(messages[0]["content"])

    async def harvest_complete(self, messages, max_tokens=None) -> str:
        self._acc(self.harvest)
        return self._respond(messages[0]["content"])

    async def harvest_complete_json(self, messages, max_tokens=None) -> dict:
        self._acc(self.harvest)
        return self._respond_json(messages[0]["content"])

    def get_usage_stats(self) -> dict:
        return {
            "primary": {
                "model": self.primary.model_name,
                "requests": self.primary.total_requests,
                "prompt_tokens": self.primary.total_prompt_tokens,
                "completion_tokens": self.primary.total_completion_tokens,
                "total_tokens": (self.primary.total_prompt_tokens
                                 + self.primary.total_completion_tokens),
            },
            "harvest": {
                "model": self.harvest.model_name,
                "requests": self.harvest.total_requests,
                "prompt_tokens": self.harvest.total_prompt_tokens,
                "completion_tokens": self.harvest.total_completion_tokens,
                "total_tokens": (self.harvest.total_prompt_tokens
                                 + self.harvest.total_completion_tokens),
            },
        }

    def _respond_json(self, prompt: str) -> dict:
        # Peer-Review Planning
        if "Peer Review" in prompt and "criteria" in prompt:
            return {
                "paper_type": "article",
                "criteria": ["Methodik", "Relevanz", "Klarheit"],
                "target_audience": "reviewer",
            }
        # criterion evaluation
        if "score" in prompt.lower() or "bewertung" in prompt.lower():
            return {
                "score": 4,
                "rationale": "Mock rationale",
                "strengths": ["ok"],
                "weaknesses": [],
            }
        return {}

    def _respond(self, prompt: str) -> str:
        if "finaler" in prompt.lower() or "synthese" in prompt.lower():
            return "# Peer Review Mock\n\nMock report body."
        return "Mock response text."


# ─── Test-Helpers ──────────────────────────────────────────────────


def _preflight_inputs_peer_review():
    # Peer review expects at least 2000 characters of paper_text + discipline + role
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


async def _new_runner():
    mock_llm = SmartMockLLM()
    return AnalysisPipelineRunner(
            progress_callback=None,
        llm_client=mock_llm, max_parallel=3, rate_limit=100,
    )


async def _new_context(use_case: str, inputs: dict) -> HarvestContext:
    ctx = HarvestContext(query=inputs.get("paper_text", "test")[:80])
    ctx.use_case = use_case
    ctx.preflight_data = inputs
    return ctx


# ─── Tests ─────────────────────────────────────────────────────────


async def test_decompose_only_returns_plan():
    """decompose_only returns a plan without executing it."""
    print("Test: decompose_only returns a plan ...", end=" ")
    runner = await _new_runner()
    ctx = await _new_context("peer_review", _preflight_inputs_peer_review())
    ctx._llm_client = runner.llm_client

    plan = (await runner.decompose_only(ctx)).task_plan

    assert plan is not None, f"plan is None: {ctx.error_message}"
    assert plan.use_case == "peer_review"
    assert len(plan.tasks) > 0
    # task_plan must be in ctx too
    assert ctx.task_plan is plan
    # The pipeline status is not "done" yet — nothing was executed
    assert ctx.status != "done"
    # no task results
    assert not ctx.task_results or len(ctx.task_results) == 0
    print(f"✓ ({len(plan.tasks)} Tasks, {plan.estimated_calls} Calls)")


async def test_decompose_only_then_run_skip_decompose():
    """Two-phase flow: decompose_only → run(skip_decompose=True) returns
    a complete result."""
    print("Test: decompose_only + run(skip_decompose) ...", end=" ")
    runner = await _new_runner()
    ctx = await _new_context("peer_review", _preflight_inputs_peer_review())
    ctx._llm_client = runner.llm_client

    # Phase 1: only build the plan
    plan = (await runner.decompose_only(ctx)).task_plan
    assert plan is not None

    # Phase 2: run the plan
    await runner.run(ctx, skip_decompose=True)

    assert ctx.status == "done", (
        f"status: {ctx.status}, error: {ctx.error_message}"
    )
    assert ctx.final_report, "final_report is empty"
    # the task results must be there now
    assert len(ctx.task_results) > 0
    # the task plan is the same as after decompose_only
    assert ctx.task_plan is plan
    print(f"✓ ({len(ctx.task_results)} Tasks done)")


async def test_skip_decompose_without_plan_fails():
    """run(skip_decompose=True) without a previous ctx.task_plan returns
    an error."""
    print("Test: skip_decompose without a plan → error ...", end=" ")
    runner = await _new_runner()
    ctx = await _new_context("peer_review", _preflight_inputs_peer_review())
    ctx._llm_client = runner.llm_client
    # no decompose_only() — ctx.task_plan is None

    await runner.run(ctx, skip_decompose=True)

    assert ctx.status == "error"
    assert "TaskPlan" in (ctx.error_message or "")
    print("✓")


async def test_normal_run_still_works():
    """A normal run() without skip_decompose works unchanged."""
    print("Test: normal run() unchanged ...", end=" ")
    runner = await _new_runner()
    ctx = await _new_context("peer_review", _preflight_inputs_peer_review())
    ctx._llm_client = runner.llm_client

    await runner.run(ctx)

    assert ctx.status == "done", f"Status: {ctx.status}: {ctx.error_message}"
    assert ctx.final_report
    assert len(ctx.task_results) > 0
    print("✓")


async def test_decompose_only_result_matches_full_run():
    """The plan from decompose_only is structurally identical to the one a
    normal run() creates."""
    print("Test: decompose_only Plan == full run Plan ...", end=" ")

    runner_a = await _new_runner()
    ctx_a = await _new_context(
        "peer_review", _preflight_inputs_peer_review()
    )
    ctx_a._llm_client = runner_a.llm_client
    plan_a = (await runner_a.decompose_only(ctx_a)).task_plan

    runner_b = await _new_runner()
    ctx_b = await _new_context(
        "peer_review", _preflight_inputs_peer_review()
    )
    ctx_b._llm_client = runner_b.llm_client
    await runner_b.run(ctx_b)

    # same number of tasks and same use case
    assert len(plan_a.tasks) == len(ctx_b.task_plan.tasks)
    assert plan_a.use_case == ctx_b.task_plan.use_case
    # same task IDs
    ids_a = sorted(t.id for t in plan_a.tasks)
    ids_b = sorted(t.id for t in ctx_b.task_plan.tasks)
    assert ids_a == ids_b, f"task IDs differ: {ids_a} vs {ids_b}"
    print("✓")


async def test_two_phase_vs_single_phase_same_result():
    """The two-phase flow produces the same number of completed tasks as
    the single-phase flow."""
    print("Test: two vs. one phase, same result ...", end=" ")

    # single-phase
    runner_single = await _new_runner()
    ctx_single = await _new_context(
        "peer_review", _preflight_inputs_peer_review()
    )
    ctx_single._llm_client = runner_single.llm_client
    await runner_single.run(ctx_single)

    # two-phase
    runner_two = await _new_runner()
    ctx_two = await _new_context(
        "peer_review", _preflight_inputs_peer_review()
    )
    ctx_two._llm_client = runner_two.llm_client
    await runner_two.decompose_only(ctx_two)
    await runner_two.run(ctx_two, skip_decompose=True)

    assert ctx_single.status == "done"
    assert ctx_two.status == "done"
    # same number of done tasks
    done_single = sum(
        1 for r in ctx_single.task_results.values()
        if r.state == TaskState.DONE
    )
    done_two = sum(
        1 for r in ctx_two.task_results.values()
        if r.state == TaskState.DONE
    )
    assert done_single == done_two, (
        f"single-phase: {done_single} done, two-phase: {done_two} done"
    )
    # same number of tasks in the plan
    assert len(ctx_single.task_plan.tasks) == len(ctx_two.task_plan.tasks)
    print(f"✓ (both {done_single} done tasks)")


async def main():
    print("=" * 60)
    print("Plan-Gate: Runner-Split Tests")
    print("=" * 60)

    tests = [
        test_decompose_only_returns_plan,
        test_decompose_only_then_run_skip_decompose,
        test_skip_decompose_without_plan_fails,
        test_normal_run_still_works,
        test_decompose_only_result_matches_full_run,
        test_two_phase_vs_single_phase_same_result,
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
