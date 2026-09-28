"""
Tests for `_filter_off_topic_sources`.

Acceptance:
  - the pattern heuristic `_is_listing_page` stays as a first stage.
  - the classifier `judge_source_relevance` complements it semantically:
    what the heuristic misses, the classifier checks.
  - a use case can disable the classifier.
  - on an LLM error or without a plan, all sources are kept
    (conservative default — nothing is lost silently).

The tests run without booting the full orchestrator, by calling the method
via `__class__` binding on a stub.
"""

import unittest
from unittest.mock import MagicMock

from src.pipeline.models import (
    ResearchPlan,
    ResearchQuestion,
    SourceDocument,
    SourceType,
)
from tests._helpers import MockLLM, async_test


def _make_orchestrator_stub(mode: str = "web_research"):
    from src.pipeline.orchestrator import ResearchOrchestrator
    stub = MagicMock(spec=ResearchOrchestrator)
    stub._mode = mode
    stub._classifier_calls = []
    stub.llm = MagicMock()
    return stub


def _make_source(url: str, title: str, content: str = "x" * 600):
    return SourceDocument(SourceType.WEB_PAGE, url, title, content)


def _patch_adapter_with(llm: MockLLM):
    """Patch HarvestModelAdapter so that it uses the MockLLM."""
    from src.llm import classifier_adapter

    class _Pass:
        def __init__(self, _dual):
            pass
        async def complete(self, messages, max_tokens=None):
            return await llm.complete(messages, max_tokens=max_tokens)

    original = classifier_adapter.HarvestModelAdapter
    classifier_adapter.HarvestModelAdapter = _Pass
    return original  # gives the test the possibility to patch back


def _restore_adapter(original):
    from src.llm import classifier_adapter
    classifier_adapter.HarvestModelAdapter = original


class TestOffTopicFilter(unittest.TestCase):

    @async_test
    async def test_off_topic_with_high_confidence_filtered(self):
        """off_topic with conf>0.7 is excluded."""
        from src.pipeline.orchestrator import ResearchOrchestrator

        # 3 sources: 2 relevant, 1 clearly off topic
        sources = [
            _make_source("https://nvidia.com/datasheet", "DGX B300 Datasheet"),
            _make_source("https://servethehome.com/dgx-review", "DGX B300 Review"),
            _make_source("https://example.com/cooking", "Recipes for Pasta"),
        ]
        plan = ResearchPlan(
            questions=[
                ResearchQuestion(id="F1", question="DGX B300 Stromaufnahme"),
            ],
            summary="",
        )

        # LLM answer: only item 3 is off topic with high confidence
        llm = MockLLM().queue_json([
            {"index": 1, "relevance": "high", "confidence": 0.9,
             "reasoning": "Datasheet ist Primärquelle"},
            {"index": 2, "relevance": "high", "confidence": 0.85,
             "reasoning": "Review enthält Power-Tests"},
            {"index": 3, "relevance": "off_topic", "confidence": 0.95,
             "reasoning": "Kochrezepte, kein Bezug zu Hardware"},
        ])

        original = _patch_adapter_with(llm)
        try:
            stub = _make_orchestrator_stub()
            progress_cb = MagicMock(return_value=None)

            async def progress(*a, **kw):
                progress_cb(*a, **kw)

            kept = await ResearchOrchestrator._filter_off_topic_sources(
                stub, sources, plan, progress,
            )
        finally:
            _restore_adapter(original)

        self.assertEqual(len(kept), 2)
        kept_urls = {s.url for s in kept}
        self.assertIn("https://nvidia.com/datasheet", kept_urls)
        self.assertIn("https://servethehome.com/dgx-review", kept_urls)
        self.assertNotIn("https://example.com/cooking", kept_urls)

    @async_test
    async def test_low_confidence_off_topic_kept(self):
        """Off topic with low confidence stays — safe default."""
        from src.pipeline.orchestrator import ResearchOrchestrator

        sources = [_make_source("https://x.com/page", "Vielleicht relevant")]
        plan = ResearchPlan(
            questions=[ResearchQuestion(id="F1", question="x")],
            summary="",
        )

        llm = MockLLM().queue_json([
            {"index": 1, "relevance": "off_topic", "confidence": 0.5,
             "reasoning": "Bin mir nicht ganz sicher"},
        ])

        original = _patch_adapter_with(llm)
        try:
            stub = _make_orchestrator_stub()
            kept = await ResearchOrchestrator._filter_off_topic_sources(
                stub, sources, plan,
                lambda *a, **kw: None,
            )
        finally:
            _restore_adapter(original)

        # kept — the classifier is not sure
        self.assertEqual(len(kept), 1)

    @async_test
    async def test_no_questions_no_filtering(self):
        """Plan without questions → no classifier call, all sources stay."""
        from src.pipeline.orchestrator import ResearchOrchestrator

        sources = [_make_source("https://x.com", "x")]
        plan = ResearchPlan(questions=[], summary="")

        stub = _make_orchestrator_stub()
        kept = await ResearchOrchestrator._filter_off_topic_sources(
            stub, sources, plan, lambda *a, **kw: None,
        )

        self.assertEqual(kept, sources)

    @async_test
    async def test_empty_sources(self):
        from src.pipeline.orchestrator import ResearchOrchestrator
        stub = _make_orchestrator_stub()
        plan = ResearchPlan(
            questions=[ResearchQuestion(id="F1", question="x")],
            summary="",
        )

        kept = await ResearchOrchestrator._filter_off_topic_sources(
            stub, [], plan, lambda *a, **kw: None,
        )
        self.assertEqual(kept, [])

    @async_test
    async def test_llm_failure_keeps_all(self):
        """LLM failure → all sources stay (conservative default)."""
        from src.pipeline.orchestrator import ResearchOrchestrator

        sources = [_make_source(f"https://x{i}.com", "x") for i in range(3)]
        plan = ResearchPlan(
            questions=[ResearchQuestion(id="F1", question="x")],
            summary="",
        )

        from src.llm import classifier_adapter

        class _BrokenAdapter:
            def __init__(self, _dual): pass
            async def complete(self, messages, max_tokens=None):
                raise RuntimeError("LLM down")

        # Since the classifier itself (judge_source_relevance) catches the
        # LLM failure and assigns 'medium' as the default, the WRAPPER path
        # in _filter_off_topic_sources should be robust too. The batched
        # classifier does a try/except in every batch.
        original = classifier_adapter.HarvestModelAdapter
        classifier_adapter.HarvestModelAdapter = _BrokenAdapter
        try:
            stub = _make_orchestrator_stub()
            kept = await ResearchOrchestrator._filter_off_topic_sources(
                stub, sources, plan, lambda *a, **kw: None,
            )
        finally:
            classifier_adapter.HarvestModelAdapter = original

        # the classifier-internal fallback gives everything 'medium' →
        # all sources stay
        self.assertEqual(len(kept), 3)

    @async_test
    async def test_classifier_calls_logged(self):
        """Classifier calls are recorded in the stub attribute."""
        from src.pipeline.orchestrator import ResearchOrchestrator

        sources = [_make_source(f"https://x{i}.com", "t") for i in range(3)]
        plan = ResearchPlan(
            questions=[ResearchQuestion(id="F1", question="x")],
            summary="",
        )

        llm = MockLLM().queue_json([
            {"index": i + 1, "relevance": "medium", "confidence": 0.6,
             "reasoning": "x"}
            for i in range(3)
        ])

        original = _patch_adapter_with(llm)
        try:
            stub = _make_orchestrator_stub()
            stub._classifier_calls = []  # fresh
            await ResearchOrchestrator._filter_off_topic_sources(
                stub, sources, plan, lambda *a, **kw: None,
            )
        finally:
            _restore_adapter(original)

        # at least one classifier call was logged
        self.assertGreater(len(stub._classifier_calls), 0)


if __name__ == "__main__":
    unittest.main()
