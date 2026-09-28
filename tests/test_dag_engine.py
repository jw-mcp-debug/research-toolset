"""
Tests for the DAG engine.

Acceptance:
  - nodes run in order
  - applies_to=False → the node is skipped, the pipeline continues
  - STOP_PIPELINE → the remaining nodes do not run
  - FAILED + fail_mode=CONTINUE → the pipeline continues
  - FAILED + fail_mode=STOP → the exception propagates
  - StopSignal → CancelledError before every node
  - node_results are persisted in ctx
"""

import asyncio
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

from src.core.stop_signal import StopSignal
from src.pipeline.dag import (
    FailMode,
    NodeResult,
    NodeStatus,
    PipelineDAG,
    PipelineNode,
)
from tests._helpers import async_test


@dataclass
class _Ctx:
    """Minimal context for engine tests."""
    log: list = field(default_factory=list)
    node_results: list = field(default_factory=list)


class _RecordingNode(PipelineNode):
    """Node that records its run in ctx.log."""
    def __init__(self, name, applies=True, raises=None,
                 returns=None, fail_mode=FailMode.CONTINUE,
                 sleep=0.0):
        self.name = name
        self.fail_mode = fail_mode
        self._applies = applies
        self._raises = raises
        self._returns = returns
        self._sleep = sleep

    def applies_to(self, ctx):
        return self._applies, "" if self._applies else f"{self.name} skipped"

    async def run(self, ctx):
        if self._sleep:
            await asyncio.sleep(self._sleep)
        ctx.log.append(self.name)
        if self._raises:
            raise self._raises
        return self._returns


class TestPipelineDAGEngine(unittest.TestCase):

    @async_test
    async def test_runs_nodes_in_order(self):
        ctx = _Ctx()
        pipeline = PipelineDAG([
            _RecordingNode("A"),
            _RecordingNode("B"),
            _RecordingNode("C"),
        ])
        await pipeline.run(ctx)
        self.assertEqual(ctx.log, ["A", "B", "C"])

    @async_test
    async def test_skipped_node_does_not_run(self):
        ctx = _Ctx()
        pipeline = PipelineDAG([
            _RecordingNode("A"),
            _RecordingNode("B", applies=False),
            _RecordingNode("C"),
        ])
        await pipeline.run(ctx)
        self.assertEqual(ctx.log, ["A", "C"])
        # node_results have an entry for B as skipped, though
        statuses = [r.status for r in ctx.node_results]
        self.assertEqual(
            statuses,
            [NodeStatus.OK, NodeStatus.SKIPPED, NodeStatus.OK],
        )

    @async_test
    async def test_stop_pipeline_aborts_remaining(self):
        ctx = _Ctx()
        pipeline = PipelineDAG([
            _RecordingNode("A"),
            _RecordingNode("B", returns=NodeResult(
                status=NodeStatus.STOP_PIPELINE, node_name="B",
            )),
            _RecordingNode("C"),  # must NOT run
        ])
        await pipeline.run(ctx)
        self.assertEqual(ctx.log, ["A", "B"])
        self.assertNotIn("C", ctx.log)

    @async_test
    async def test_failed_node_continue(self):
        """fail_mode=CONTINUE: the pipeline continues, the error is logged."""
        ctx = _Ctx()
        pipeline = PipelineDAG([
            _RecordingNode("A"),
            _RecordingNode("B", raises=RuntimeError("boom"),
                           fail_mode=FailMode.CONTINUE),
            _RecordingNode("C"),  # MUST run
        ])
        await pipeline.run(ctx)
        self.assertEqual(ctx.log, ["A", "B", "C"])

        # B has status FAILED
        b_result = ctx.node_results[1]
        self.assertEqual(b_result.status, NodeStatus.FAILED)
        self.assertIsNotNone(b_result.error)
        self.assertIn("boom", str(b_result.error))

    @async_test
    async def test_failed_node_stop(self):
        """fail_mode=STOP: the exception propagates."""
        ctx = _Ctx()
        pipeline = PipelineDAG([
            _RecordingNode("A"),
            _RecordingNode("B", raises=RuntimeError("kritisch"),
                           fail_mode=FailMode.STOP),
            _RecordingNode("C"),
        ])
        with self.assertRaises(RuntimeError) as ctxmgr:
            await pipeline.run(ctx)
        self.assertIn("kritisch", str(ctxmgr.exception))
        # A ran, B crashed, C did not
        self.assertEqual(ctx.log, ["A", "B"])

    @async_test
    async def test_stop_signal_check_before_each_node(self):
        """Stop-signal check BEFORE every node — takes effect between nodes already."""
        signal = StopSignal()
        ctx = _Ctx()

        # Stop after the first node
        class _StoppingNode(_RecordingNode):
            async def run(self, ctx):
                ctx.log.append(self.name)
                signal.request_stop("test")
                return None

        pipeline = PipelineDAG([
            _StoppingNode("A"),
            _RecordingNode("B"),  # must NOT run
        ])

        with self.assertRaises(asyncio.CancelledError):
            await pipeline.run(ctx, stop_signal=signal)

        self.assertEqual(ctx.log, ["A"])

    @async_test
    async def test_progress_callback_emits_events(self):
        ctx = _Ctx()
        events = []

        async def cb(event_type, data):
            events.append((event_type, data))

        pipeline = PipelineDAG([
            _RecordingNode("A"),
            _RecordingNode("B", applies=False),
            _RecordingNode("C", raises=ValueError("oops")),
        ])
        await pipeline.run(ctx, progress_callback=cb)

        # Events: node_start for A and C, node_done for A,
        # node_failed for C, NO start for B (because skipped)
        event_types = [e[0] for e in events]
        self.assertIn("node_start", event_types)
        self.assertIn("node_done", event_types)
        self.assertIn("node_failed", event_types)

    @async_test
    async def test_node_records_duration(self):
        """The engine measures the node duration."""
        ctx = _Ctx()
        pipeline = PipelineDAG([
            _RecordingNode("Slow", sleep=0.02),
        ])
        await pipeline.run(ctx)
        # the node took at least 20 ms
        self.assertGreaterEqual(
            ctx.node_results[0].duration_seconds, 0.02,
        )

    @async_test
    async def test_dict_return_becomes_metadata(self):
        ctx = _Ctx()
        pipeline = PipelineDAG([
            _RecordingNode("A", returns={"foo": "bar", "n": 42}),
        ])
        await pipeline.run(ctx)
        self.assertEqual(
            ctx.node_results[0].metadata,
            {"foo": "bar", "n": 42},
        )

    @async_test
    async def test_empty_pipeline(self):
        """0 nodes — a clean no-op."""
        ctx = _Ctx()
        pipeline = PipelineDAG([])
        await pipeline.run(ctx)
        self.assertEqual(ctx.log, [])
        self.assertEqual(ctx.node_results, [])


class TestPipelineNodeBaseClass(unittest.TestCase):

    @async_test
    async def test_default_applies_is_true(self):
        node = PipelineNode()
        applies, reason = node.applies_to(SimpleNamespace())
        self.assertTrue(applies)

    @async_test
    async def test_default_run_raises(self):
        node = PipelineNode()
        with self.assertRaises(NotImplementedError):
            await node.run(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
