"""
Tests for the DAG executor.

Uses a mock LLM that returns deterministic answers. Covers:
- linear pipeline
- parallel pipeline
- mixed DAG
- retry with success
- retries exhausted → FAILED
- dependency skip
- resume from a checkpoint
- rate-limiter delay
"""

import asyncio
import sys
import time
from pathlib import Path

# path to the source file (tests run from tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.dag_executor import (
    DAGExecutor,
    RateLimitException,
)
from src.pipeline.models import (
    HarvestContext,
    SubTask,
    TaskPlan,
    TaskState,
)


# ─── Mock-LLM ──────────────────────────────────────────────────────

class MockLLM:
    """Deterministic mock LLM for tests.

    Behaviour configurable per task_id:
    - "success" (default): returns dummy output immediately
    - "fail_then_succeed": fails on the first attempt, succeeds on the second
    - "always_fail": always raises
    - "rate_limit_then_succeed": once RateLimit, then success
    - "delay:N": wait N seconds, then success
    - "json": returns JSON output
    """

    def __init__(self, behavior: dict[str, str] = None):
        self.behavior = behavior or {}
        self.call_log: list[tuple[str, int]] = []  # (task_id, attempt)
        self.call_counts: dict[str, int] = {}

    async def call(self, prompt: str, meta: dict) -> tuple[str, dict]:
        """Mock implementation of the LLM interface."""
        task_id = meta.get("task_id", "unknown")
        self.call_counts[task_id] = self.call_counts.get(task_id, 0) + 1
        attempt = self.call_counts[task_id]
        self.call_log.append((task_id, attempt))

        behavior = self.behavior.get(task_id, "success")

        if behavior == "success":
            return f"Output for {task_id}", {"tokens_used": 100}

        if behavior == "always_fail":
            raise RuntimeError(f"mock error for {task_id}")

        if behavior == "fail_then_succeed":
            if attempt == 1:
                raise RuntimeError(f"let the first attempt fail ({task_id})")
            return f"Output for {task_id} nach Retry", {"tokens_used": 100}

        if behavior == "rate_limit_then_succeed":
            if attempt == 1:
                raise RateLimitException(f"429 for {task_id}")
            return f"Output for {task_id} nach Rate-Limit", {"tokens_used": 100}

        if behavior.startswith("delay:"):
            seconds = float(behavior.split(":")[1])
            await asyncio.sleep(seconds)
            return f"Output for {task_id} nach {seconds}s", {"tokens_used": 100}

        if behavior == "json":
            return '```json\n{"result": "ok", "value": 42}\n```', {"tokens_used": 100}

        return f"Default output for {task_id}", {"tokens_used": 100}


# ─── Test helpers ───────────────────────────────────────────────────

def make_context() -> HarvestContext:
    """Create an empty HarvestContext."""
    return HarvestContext(query="test")


def make_task(task_id: str, deps: list[str] = None, max_retries: int = 0) -> SubTask:
    """Helper for creating a SubTask."""
    return SubTask(
        id=task_id,
        phase="test",
        description=f"Test {task_id}",
        prompt_template="dummy",
        prompt_params={"task_id": task_id},
        depends_on=deps or [],
        max_retries=max_retries,
    )


PROMPTS = {"dummy": "Task: {task_id}"}


# ─── Tests ─────────────────────────────────────────────────────────

async def test_linear_pipeline():
    """3 tasks in sequence: A → B → C."""
    print("Test: linear pipeline ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task("A"),
        make_task("B", deps=["A"]),
        make_task("C", deps=["B"]),
    ])
    mock = MockLLM()
    executor = DAGExecutor(llm_call=mock.call, max_parallel=5, rate_limit=100)
    ctx = make_context()
    results = await executor.execute(plan, PROMPTS, ctx)

    assert len(results) == 3
    assert all(r.state == TaskState.DONE for r in results.values())
    assert ctx.completed_layers == [0, 1, 2]
    # linear order: A before B before C
    a_time = results["A"].finished_at
    b_time = results["B"].started_at
    assert a_time <= b_time, f"A finished at {a_time}, B started at {b_time}"
    print("✓")


async def test_parallel_layer():
    """5 tasks in parallel in one layer."""
    print("Test: parallel layer ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task(f"T{i}") for i in range(5)
    ])
    mock = MockLLM({f"T{i}": "delay:0.1" for i in range(5)})
    executor = DAGExecutor(llm_call=mock.call, max_parallel=5, rate_limit=100)
    ctx = make_context()

    start = time.monotonic()
    await executor.execute(plan, PROMPTS, ctx)
    elapsed = time.monotonic() - start

    # 5 parallel tasks of 0.1 s each should run in ~0.1-0.3 s, not 0.5 s
    assert elapsed < 0.4, f"parallelism does not work: {elapsed:.2f}s"
    assert all(r.state == TaskState.DONE for r in ctx.task_results.values())
    print(f"✓ ({elapsed:.2f}s for 5 tasks of 0.1s)")


async def test_max_parallel_limit():
    """With max_parallel=2, 4 tasks of 0.1 s each should take ~0.2 s, not 0.1 s."""
    print("Test: max_parallel-Limit ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task(f"T{i}") for i in range(4)
    ])
    mock = MockLLM({f"T{i}": "delay:0.1" for i in range(4)})
    executor = DAGExecutor(llm_call=mock.call, max_parallel=2, rate_limit=100)
    ctx = make_context()

    start = time.monotonic()
    await executor.execute(plan, PROMPTS, ctx)
    elapsed = time.monotonic() - start

    # 4 tasks with at most 2 in parallel of 0.1 s each = ~0.2 s
    assert 0.15 < elapsed < 0.35, f"max_parallel=2 does not hold: {elapsed:.2f}s"
    print(f"✓ ({elapsed:.2f}s)")


async def test_mixed_dag():
    """A → (B, C) → D — a DAG with branching."""
    print("Test: mixed DAG (A → B,C → D) ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task("A"),
        make_task("B", deps=["A"]),
        make_task("C", deps=["A"]),
        make_task("D", deps=["B", "C"]),
    ])
    mock = MockLLM()
    executor = DAGExecutor(llm_call=mock.call, max_parallel=5, rate_limit=100)
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    assert all(r.state == TaskState.DONE for r in ctx.task_results.values())
    assert ctx.completed_layers == [0, 1, 2]  # 3 layers
    # B and C after A
    assert ctx.task_results["B"].started_at >= ctx.task_results["A"].finished_at
    # D after B and C
    assert ctx.task_results["D"].started_at >= ctx.task_results["B"].finished_at
    assert ctx.task_results["D"].started_at >= ctx.task_results["C"].finished_at
    print("✓")


async def test_retry_success():
    """The task fails on the first attempt, succeeds on the second."""
    print("Test: retry with success ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task("R", max_retries=2),
    ])
    mock = MockLLM({"R": "fail_then_succeed"})
    executor = DAGExecutor(llm_call=mock.call, max_parallel=5, rate_limit=100)
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    result = ctx.task_results["R"]
    assert result.state == TaskState.DONE, f"State: {result.state}"
    assert result.retries_used == 1, f"Retries: {result.retries_used}"
    assert mock.call_counts["R"] == 2
    print("✓")


async def test_retry_exhausted():
    """The task always fails, the retries are used up."""
    print("Test: retries exhausted → FAILED ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task("F", max_retries=2),
    ])
    mock = MockLLM({"F": "always_fail"})
    executor = DAGExecutor(llm_call=mock.call, max_parallel=5, rate_limit=100)
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    result = ctx.task_results["F"]
    assert result.state == TaskState.FAILED
    assert "mock error" in result.error
    assert mock.call_counts["F"] == 3  # 1 + 2 retries
    print("✓")


async def test_dependency_skip():
    """The task is SKIPPED if a dependency FAILED."""
    print("Test: dependency skip ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task("A"),
        make_task("B", deps=["A"]),  # depends on A
        make_task("C", deps=["B"]),  # depends on B
    ])
    mock = MockLLM({"A": "always_fail"})
    executor = DAGExecutor(llm_call=mock.call, max_parallel=5, rate_limit=100)
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    assert ctx.task_results["A"].state == TaskState.FAILED
    assert ctx.task_results["B"].state == TaskState.SKIPPED
    assert ctx.task_results["C"].state == TaskState.SKIPPED
    # B and C must not have been called
    assert "B" not in mock.call_counts
    assert "C" not in mock.call_counts
    print("✓")


async def test_rate_limit_recovery():
    """RateLimitException → back-off → retry → success."""
    print("Test: Rate-Limit Recovery ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task("RL", max_retries=2),
    ])
    mock = MockLLM({"RL": "rate_limit_then_succeed"})
    executor = DAGExecutor(
        llm_call=mock.call,
        max_parallel=5,
        rate_limit=100,
    )
    # shorten the back-off for the test
    executor.rate_limiter.backoff_initial = 0.05
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    result = ctx.task_results["RL"]
    assert result.state == TaskState.DONE
    assert mock.call_counts["RL"] == 2
    print("✓")


async def test_resume_from_layer():
    """Resume: the first layer is skipped, earlier results stay."""
    print("Test: resume from a checkpoint ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task("A"),
        make_task("B", deps=["A"]),
    ])
    # first run normally
    mock1 = MockLLM()
    executor1 = DAGExecutor(llm_call=mock1.call, max_parallel=5, rate_limit=100)
    ctx = make_context()
    await executor1.execute(plan, PROMPTS, ctx)
    assert mock1.call_counts == {"A": 1, "B": 1}
    assert ctx.completed_layers == [0, 1]

    # Now resume from layer 1 (i.e. only B anew, keep A)
    # a new executor with a different mock, so that we can measure who is called
    mock2 = MockLLM()
    executor2 = DAGExecutor(llm_call=mock2.call, max_parallel=5, rate_limit=100)
    await executor2.execute(plan, PROMPTS, ctx, resume_from_layer=1)

    # A must not have been called again
    assert "A" not in mock2.call_counts, f"A was called unexpectedly: {mock2.call_counts}"
    # B is run again
    assert mock2.call_counts == {"B": 1}, f"Mock2 calls: {mock2.call_counts}"
    print("✓")


async def test_progress_callback():
    """Progress events are sent correctly."""
    print("Test: Progress-Callback ...", end=" ")
    events = []

    async def cb(event: str, data: dict):
        events.append((event, data))

    plan = TaskPlan(use_case="test", tasks=[
        make_task("A"),
        make_task("B", deps=["A"]),
    ])
    mock = MockLLM()
    executor = DAGExecutor(
        llm_call=mock.call, max_parallel=5, rate_limit=100,
        progress_callback=cb,
    )
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    event_types = [e[0] for e in events]
    assert "layer_start" in event_types
    assert "task_start" in event_types
    assert "task_done" in event_types
    assert "layer_done" in event_types
    print(f"✓ ({len(events)} Events)")


async def test_persist_callback():
    """The persist callback is called after every layer."""
    print("Test: Persist-Callback ...", end=" ")
    persist_calls = []

    async def persist(ctx: HarvestContext, layer_idx: int):
        persist_calls.append((layer_idx, len(ctx.task_results)))

    plan = TaskPlan(use_case="test", tasks=[
        make_task("A"),
        make_task("B", deps=["A"]),
        make_task("C", deps=["B"]),
    ])
    mock = MockLLM()
    executor = DAGExecutor(
        llm_call=mock.call, max_parallel=5, rate_limit=100,
        persist_callback=persist,
    )
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    assert len(persist_calls) == 3  # 3 layers
    assert [c[0] for c in persist_calls] == [0, 1, 2]
    print("✓")


async def test_json_parsing():
    """JSON output is parsed automatically."""
    print("Test: JSON-Parsing ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[make_task("J")])
    mock = MockLLM({"J": "json"})
    executor = DAGExecutor(llm_call=mock.call, max_parallel=5, rate_limit=100)
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    result = ctx.task_results["J"]
    assert result.state == TaskState.DONE
    assert result.parsed_output == {"result": "ok", "value": 42}
    print("✓")


async def test_invalid_max_parallel():
    """The constructor rejects an invalid max_parallel."""
    print("Test: Invalid max_parallel ...", end=" ")
    mock = MockLLM()
    try:
        DAGExecutor(llm_call=mock.call, max_parallel=0)
        assert False, "should raise an exception"
    except ValueError:
        pass
    try:
        DAGExecutor(llm_call=mock.call, max_parallel=20)
        assert False, "should raise an exception"
    except ValueError:
        pass
    print("✓")


async def test_stop_check_between_layers():
    """A stop check between layers aborts the pipeline cleanly."""
    print("Test: stop between layers ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task("A"),
        make_task("B", deps=["A"]),
        make_task("C", deps=["B"]),
    ])

    # stop flag that becomes active after the first layer
    stop_flag = {"stop": False}

    async def stop_after_first(event: str, data: dict):
        if event == "layer_done" and data.get("layer") == 1:
            stop_flag["stop"] = True

    mock = MockLLM()
    executor = DAGExecutor(
        llm_call=mock.call,
        max_parallel=5,
        rate_limit=100,
        progress_callback=stop_after_first,
        stop_check=lambda: stop_flag["stop"],
    )
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    # A must have run
    assert ctx.task_results["A"].state == TaskState.DONE
    # B and C must be SKIPPED
    assert ctx.task_results["B"].state == TaskState.SKIPPED, \
        f"B state: {ctx.task_results['B'].state}"
    assert ctx.task_results["C"].state == TaskState.SKIPPED
    # B must not have been called
    assert "B" not in mock.call_counts
    assert "C" not in mock.call_counts
    print("✓")


async def test_stop_before_first_layer():
    """Stop before the first layer → all tasks SKIPPED."""
    print("Test: stop before the first layer ...", end=" ")
    plan = TaskPlan(use_case="test", tasks=[
        make_task("A"),
        make_task("B", deps=["A"]),
    ])
    mock = MockLLM()
    executor = DAGExecutor(
        llm_call=mock.call,
        max_parallel=5,
        rate_limit=100,
        stop_check=lambda: True,  # abort immediately
    )
    ctx = make_context()
    await executor.execute(plan, PROMPTS, ctx)

    # all tasks must be SKIPPED
    assert ctx.task_results["A"].state == TaskState.SKIPPED
    assert ctx.task_results["B"].state == TaskState.SKIPPED
    # no LLM calls
    assert not mock.call_counts
    print("✓")


# ─── Test-Runner ───────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("DAG-Executor Tests")
    print("=" * 60)
    tests = [
        test_linear_pipeline,
        test_parallel_layer,
        test_max_parallel_limit,
        test_mixed_dag,
        test_retry_success,
        test_retry_exhausted,
        test_dependency_skip,
        test_rate_limit_recovery,
        test_resume_from_layer,
        test_progress_callback,
        test_persist_callback,
        test_json_parsing,
        test_invalid_max_parallel,
        test_stop_check_between_layers,
        test_stop_before_first_layer,
    ]
    failed = 0
    for test in tests:
        try:
            await test()
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
    asyncio.run(main())
