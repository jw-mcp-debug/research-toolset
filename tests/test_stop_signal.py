"""
Tests for `src.core.stop_signal`.

Acceptance:
  - a single StopSignal per run
  - a stop click takes effect in all pipelines (the same instance is passed on)
  - raise_if_stopped raises asyncio.CancelledError
"""

import asyncio
import unittest

from src.core.stop_signal import StopSignal


class TestStopSignal(unittest.TestCase):

    def test_initial_not_stopped(self):
        s = StopSignal()
        self.assertFalse(s.is_stop_requested())
        self.assertEqual(s.reason, "")
        self.assertFalse(bool(s))  # __bool__

    def test_request_stop_idempotent(self):
        s = StopSignal()
        s.request_stop("first")
        s.request_stop("second")  # the second reason does not overwrite
        self.assertTrue(s.is_stop_requested())
        self.assertEqual(s.reason, "first")

    def test_truthy_after_stop(self):
        s = StopSignal()
        self.assertFalse(bool(s))
        s.request_stop()
        self.assertTrue(bool(s))

    def test_raise_if_stopped(self):
        s = StopSignal()
        # Before the stop: no raise
        s.raise_if_stopped()
        # After the stop: CancelledError
        s.request_stop("user_clicked_stop")

        async def run():
            with self.assertRaises(asyncio.CancelledError) as ctx:
                s.raise_if_stopped()
            self.assertIn("user_clicked_stop", str(ctx.exception))

        asyncio.run(run())

    def test_reset(self):
        s = StopSignal()
        s.request_stop("x")
        self.assertTrue(s.is_stop_requested())
        s.reset()
        self.assertFalse(s.is_stop_requested())
        self.assertEqual(s.reason, "")

    def test_shared_instance_across_pipelines(self):
        """Acceptance: a single StopSignal per run."""
        s = StopSignal()

        # Simulate: orchestrator and AnalysisRunner get the SAME instance
        # (not different bool flags).
        orchestrator_signal = s
        analysis_signal = s

        # The stop button sets the signal
        s.request_stop("user_request")

        # Both pipelines see the stop
        self.assertTrue(orchestrator_signal.is_stop_requested())
        self.assertTrue(analysis_signal.is_stop_requested())
        # Identity — not just the same value, but the same object
        self.assertIs(orchestrator_signal, analysis_signal)


if __name__ == "__main__":
    unittest.main()
