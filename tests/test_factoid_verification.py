"""
Tests for `_verify_report_factoids` and `_annotate_unverified_factoids`.

Acceptance:
  - factoids are extracted from the report
  - verified against positive extracts
  - refuted factoids (verified='false') are appended to the report
  - 'uncertain' is NOT marked inline (too many false positives)
  - without extracts: all factoids stay uncertain, no crash
  - on LLM failure: graceful, the report stays unchanged
"""

import unittest
from unittest.mock import MagicMock

from tests._helpers import MockLLM


def _make_orchestrator_stub(mode: str = "web_research"):
    from src.pipeline.orchestrator import ResearchOrchestrator
    stub = MagicMock(spec=ResearchOrchestrator)
    stub._mode = mode
    stub._classifier_calls = []
    stub.llm = MagicMock()
    return stub


def _patch_adapter_with(llm: MockLLM):
    from src.llm import classifier_adapter

    class _Pass:
        def __init__(self, _dual): pass
        async def complete(self, messages, max_tokens=None):
            return await llm.complete(messages, max_tokens=max_tokens)

    original = classifier_adapter.HarvestModelAdapter
    classifier_adapter.HarvestModelAdapter = _Pass
    return original


def _restore_adapter(original):
    from src.llm import classifier_adapter
    classifier_adapter.HarvestModelAdapter = original


class TestAnnotateUnverifiedFactoids(unittest.TestCase):
    """Test the static annotation logic in isolation."""

    def test_no_unverified_no_change(self):
        from src.pipeline.orchestrator import _annotate_unverified_factoids
        report = "## Bericht\n\nInhalt"
        verifications = [
            {"factoid": "x", "type": "claim", "verified": "true",
             "supporting_extract_id": "E1", "supporting_quote": "x",
             "confidence": 0.9, "fallback_used": False},
        ]
        out = _annotate_unverified_factoids(report, verifications)
        self.assertEqual(out, report)

    def test_only_uncertain_no_block(self):
        """'uncertain' is not marked in the report."""
        from src.pipeline.orchestrator import _annotate_unverified_factoids
        report = "## Bericht"
        verifications = [
            {"factoid": "x", "type": "claim", "verified": "uncertain",
             "supporting_extract_id": None, "supporting_quote": "",
             "confidence": 0.4, "fallback_used": True},
        ]
        out = _annotate_unverified_factoids(report, verifications)
        self.assertEqual(out, report)

    def test_low_conf_unverified_appended_to_report(self):
        """Low-confidence contradictions are listed in a discreet block at the
        end of the report (high-confidence cases go to
        ReportRevisionNode, not here)."""
        from src.pipeline.orchestrator import _annotate_unverified_factoids
        report = "## Bericht\n\nDie DGX kostet 500k USD und kam 2024."
        verifications = [
            # low confidence → goes into the notice block
            {"factoid": "DGX kostet 500k USD", "type": "numeric_spec",
             "verified": "false", "supporting_extract_id": "E1",
             "supporting_quote": "Listenpreis: $415k",
             "confidence": 0.65, "fallback_used": False},
            # verified → does not go there
            {"factoid": "kam 2024", "type": "date",
             "verified": "true", "supporting_extract_id": "E2",
             "supporting_quote": "released October 2024",
             "confidence": 0.95, "fallback_used": False},
        ]
        out = _annotate_unverified_factoids(report, verifications)
        # start of the report kept
        self.assertTrue(out.startswith(report))
        # notice block appended
        self.assertIn("Notes on source consistency", out)
        # only THE refuted factoid in it (true is not listed)
        self.assertIn("DGX kostet 500k USD", out.split("---")[-1])
        self.assertIn("Listenpreis: $415k", out)
        # the confidence is shown
        self.assertIn("0.65", out)

    def test_high_conf_unverified_NOT_in_report(self):
        """High-confidence contradictions (≥0.9) are NOT handled by
        _annotate_unverified_factoids — they go to ReportRevisionNode
        for a semantic correction."""
        from src.pipeline.orchestrator import _annotate_unverified_factoids
        report = "## Bericht\n\nDie DGX kostet 500k USD."
        verifications = [
            {"factoid": "DGX kostet 500k USD", "type": "numeric_spec",
             "verified": "false", "supporting_extract_id": "E1",
             "supporting_quote": "Listenpreis: $415k",
             "confidence": 0.95,  # high confidence!
             "fallback_used": False},
        ]
        out = _annotate_unverified_factoids(report, verifications)
        # the report stays completely unchanged — no annotation, no block
        self.assertEqual(out, report,
                         "high-confidence cases must no longer "
                         "be appended — they go to "
                         "ReportRevisionNode.")


if __name__ == "__main__":
    unittest.main()
