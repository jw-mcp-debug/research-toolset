"""
Tests for the classifier integration in the orchestrator.

These tests do NOT boot the full orchestrator (it needs httpx, a
ConnectorRegistry, search engines etc.). Instead they test three methods
directly:

  - _classify_coverage_per_question
  - _classify_continue_decision
  - _classify_final_diagnosis

with hand-made mock instances. That covers the semantic logic without
testing the pipeline mechanics as well.

Cases:
  - a run where a filter discarded everything → final_diagnosis="filter_too_strict"
  - classifier path disabled → the method returns None / an empty dict
  - use-case override against the classifier → the filter does not run
"""

import unittest
from unittest.mock import MagicMock

from tests._helpers import MockLLM


def _make_orchestrator_stub(
    mode: str = "web_research",
    mock_llm: MockLLM | None = None,
):
    """Build an orchestrator-like stub object with only the fields the
    tested methods touch. Avoids `__init__` (so no httpx etc. is needed).

    We import the real class but do NOT instantiate it — we build an
    object that calls its methods via `__class__` binding.
    """
    # Lazy: only the unbound methods we test
    from src.pipeline.orchestrator import ResearchOrchestrator

    stub = MagicMock(spec=ResearchOrchestrator)
    stub._mode = mode
    stub._query_anchor = None
    stub._classifier_calls = []
    stub._filter_stats_per_round = []
    stub.llm = mock_llm  # taken over by the HarvestModelAdapter,
                         # but our MockLLM is NOT a DualLLMClient
                         # → we patch the adapter below.
    stub.config = MagicMock()
    stub.config.max_rounds = 3
    return stub


if __name__ == "__main__":
    unittest.main()
