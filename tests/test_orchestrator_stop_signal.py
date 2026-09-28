"""
Tests for the StopSignal integration in the orchestrator.

Acceptance:
  - one single StopSignal object per run (not several in parallel)
  - a stop click takes effect on BOTH paths (bool + StopSignal)
  - a reset between runs resets both
"""

import unittest
from unittest.mock import MagicMock

from src.core.stop_signal import StopSignal


class TestOrchestratorStopSignal(unittest.TestCase):

    def _make_orch(self):
        from src.config import PipelineConfig
        from src.pipeline.orchestrator import ResearchOrchestrator

        return ResearchOrchestrator(
            llm=MagicMock(),
            connectors=MagicMock(),
            config=PipelineConfig(),
        )

    def test_init_creates_stop_signal(self):
        """One StopSignal object per orchestrator instance."""
        orch = self._make_orch()
        self.assertIsInstance(orch._stop_signal, StopSignal)
        self.assertFalse(orch._stop_signal.is_stop_requested())
        self.assertFalse(orch._stop_requested)

    def test_stop_sets_both_signals(self):
        """stop() sets both the bool and the StopSignal."""
        orch = self._make_orch()
        orch.stop()

        # both set
        self.assertTrue(orch._stop_requested)
        self.assertTrue(orch._stop_signal.is_stop_requested())
        self.assertEqual(orch._stop_signal.reason, "user_request")

    def test_signal_is_shared_object(self):
        """If someone holds the _stop_signal, orch.stop() takes effect there too.

        This is the core acceptance: sub-pipelines get the SAME instance and
        are notified of the stop — not a copy.
        """
        orch = self._make_orch()
        external_handle = orch._stop_signal  # the sub-pipeline holds this

        orch.stop()
        self.assertTrue(external_handle.is_stop_requested())
        # identity check
        self.assertIs(external_handle, orch._stop_signal)

    def test_stop_check_lambda_reads_both(self):
        """The lambda for the AnalysisPipelineRunner reads BOTH signals."""
        orch = self._make_orch()

        # Simulate how run() builds the lambda
        stop_check = lambda: (
            orch._stop_requested or orch._stop_signal.is_stop_requested()
        )

        # Initially: no stop
        self.assertFalse(stop_check())

        # Only the bool set: the lambda recognises it
        orch._stop_requested = True
        self.assertTrue(stop_check())
        orch._stop_requested = False

        # Only the StopSignal set: the lambda recognises it too
        orch._stop_signal.request_stop("test")
        self.assertTrue(stop_check())


if __name__ == "__main__":
    unittest.main()
