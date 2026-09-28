"""
Integration tests for the pipeline nodes of the classifier layer.

Acceptance:
  - every node can be tested in isolation (without the full orchestrator).
  - nodes can be chained into a pipeline.
  - a query that a heuristic would misread as a person runs through the
    DAG with the same behaviour as the direct classifier call.
"""

import unittest
from unittest.mock import MagicMock

from src.pipeline.classifiers.query_anchor import QueryAnchor
from src.pipeline.dag import PipelineDAG, NodeStatus
from src.pipeline.dag_nodes import (
    QueryAnchorNode,
    OffTopicFilterNode,
    CoverageNode,
    ContinueDecisionNode,
    DiagnosisNode,
    FactoidVerificationNode,
)
from src.pipeline.models import (
    HarvestContext,
    ResearchPlan,
    ResearchQuestion,
    SourceDocument,
    SourceExtract,
    SourceType,
)
from tests._helpers import MockLLM, async_test


def _patch_adapter_with(llm: MockLLM):
    from src.llm import classifier_adapter

    class _Pass:
        def __init__(self, _dual):
            pass
        async def complete(self, messages, max_tokens=None):
            return await llm.complete(messages, max_tokens=max_tokens)

    original = classifier_adapter.HarvestModelAdapter
    classifier_adapter.HarvestModelAdapter = _Pass
    return original


def _restore_adapter(original):
    from src.llm import classifier_adapter
    classifier_adapter.HarvestModelAdapter = original


def _make_ctx(query: str = "Test", with_plan: bool = True):
    ctx = HarvestContext(query=query)
    ctx.classifier_calls = []
    if with_plan:
        ctx.research_plan = ResearchPlan(
            questions=[ResearchQuestion(id="F1", question="Frage 1")],
            summary="",
        )
    return ctx


# ───────────────────────────────────────────────────────────────────
# QueryAnchorNode
# ───────────────────────────────────────────────────────────────────


class TestQueryAnchorNode(unittest.TestCase):

    @async_test
    async def test_dgx_query_classifies_none(self):
        """CRITICAL: a product-comparison query → anchor.type='none'."""
        llm = MockLLM().queue_json({
            "anchor_type": "none", "target": "", "confidence": 0.94,
            "reasoning": "Vergleichsanfrage zu Hardware-Spezifikationen ohne Personenbezug.",
        })
        original = _patch_adapter_with(llm)
        try:
            node = QueryAnchorNode(llm=MagicMock())
            ctx = _make_ctx("Vergleich DGX B300 vs B200 — Preis-Leistungs-Verhältnis")

            metadata = await node.run(ctx)
        finally:
            _restore_adapter(original)

        self.assertEqual(ctx.query_anchor.type, "none")
        self.assertFalse(ctx.query_anchor.is_person())
        self.assertEqual(metadata["anchor_type"], "none")
        self.assertEqual(len(ctx.classifier_calls), 1)

    @async_test
    async def test_skipped_if_already_classified(self):
        """If ctx.query_anchor is already set: not again."""
        node = QueryAnchorNode(llm=MagicMock())
        ctx = _make_ctx()
        ctx.query_anchor = QueryAnchor(type="person", target="X",
                                       confidence=0.9)

        applies, reason = node.applies_to(ctx)
        self.assertFalse(applies)
        self.assertIn("already set", reason)


# ───────────────────────────────────────────────────────────────────
# OffTopicFilterNode
# ───────────────────────────────────────────────────────────────────


class TestOffTopicFilterNode(unittest.TestCase):

    @async_test
    async def test_filters_off_topic_sources(self):
        llm = MockLLM().queue_json([
            {"index": 1, "relevance": "high", "confidence": 0.9,
             "reasoning": "Direkter Bezug"},
            {"index": 2, "relevance": "off_topic", "confidence": 0.95,
             "reasoning": "Komplett unverwandtes Thema"},
        ])

        original = _patch_adapter_with(llm)
        try:
            node = OffTopicFilterNode(llm=MagicMock())
            ctx = _make_ctx()
            ctx.sources = [
                SourceDocument(SourceType.WEB_PAGE,
                               "https://relevant.com", "ok",
                               "x" * 600),
                SourceDocument(SourceType.WEB_PAGE,
                               "https://cooking.com", "irrelevant",
                               "y" * 600),
            ]
            await node.run(ctx)
        finally:
            _restore_adapter(original)

        self.assertEqual(len(ctx.sources), 1)
        self.assertEqual(ctx.sources[0].url, "https://relevant.com")

    @async_test
    async def test_skipped_without_plan(self):
        node = OffTopicFilterNode(llm=MagicMock())
        ctx = _make_ctx(with_plan=False)
        applies, _ = node.applies_to(ctx)
        self.assertFalse(applies)


# ───────────────────────────────────────────────────────────────────
# CoverageNode + ContinueDecisionNode (together)
# ───────────────────────────────────────────────────────────────────


class TestCoverageAndContinueNodes(unittest.TestCase):

    @async_test
    async def test_coverage_then_continue_stops(self):
        """Coverage answered → Continue stop_done → Pipeline-Stop."""
        llm = MockLLM()
        # coverage answer
        llm.queue_json({
            "coverage": "answered", "confidence": 0.9,
            "missing_aspects": [], "supporting_extract_ids": [],
            "reasoning": "Frage ist beantwortet.",
        })
        # continue answer
        llm.queue_json({
            "decision": "stop_done", "confidence": 0.88,
            "next_round_focus": "",
            "reasoning": "Coverage hoch, weitere Runde brächte nichts.",
        })

        original = _patch_adapter_with(llm)
        try:
            ctx = _make_ctx()
            ctx.extracts = [
                SourceExtract("u", "t", "F1", "fact", polarity="positive"),
            ]
            ctx.sources = [
                SourceDocument(SourceType.WEB_PAGE, "u", "t", "c"),
            ]

            cov_node = CoverageNode(llm=MagicMock(), round_number=1)
            cont_node = ContinueDecisionNode(
                llm=MagicMock(), round_number=1, max_rounds=3,
                new_extracts=1, new_sources=1,
            )

            pipeline = PipelineDAG([cov_node, cont_node])
            await pipeline.run(ctx)
        finally:
            _restore_adapter(original)

        # coverage delivered a result
        self.assertEqual(len(ctx.coverage_per_round), 1)
        self.assertEqual(
            ctx.coverage_per_round[0]["results"]["F1"]["coverage"],
            "answered",
        )
        # continue decided stop_done
        self.assertEqual(ctx.continue_decisions[0]["decision"], "stop_done")
        # the engine processed the STOP_PIPELINE signal
        cont_result = ctx.node_results[1]
        self.assertEqual(cont_result.status, NodeStatus.STOP_PIPELINE)


# ───────────────────────────────────────────────────────────────────
# DiagnosisNode — a run in which a filter discarded everything
# ───────────────────────────────────────────────────────────────────


class TestDiagnosisNode(unittest.TestCase):

    @async_test
    async def test_dgx_diagnosis_via_dag(self):
        """CRITICAL: final state of such a run → diagnosis='filter_too_strict'.

        This time via the DAG pipeline — verifies that the classifier
        behaviour is preserved in the DAG.
        """
        llm = MockLLM().queue_json({
            "diagnosis": "filter_too_strict", "confidence": 0.95,
            "remediation": "Recherche ohne Personen-Filter neu starten",
            "user_message": "528 Extrakte vom Filter verworfen",
            "reasoning": "Filter-Verlustrate 100%, alle Fragen filter_blocked.",
        })

        original = _patch_adapter_with(llm)
        try:
            ctx = _make_ctx("DGX B300 vs B200")
            ctx.research_plan = ResearchPlan(
                questions=[
                    ResearchQuestion(id=f"F{i}", question="x", priority="hoch")
                    for i in range(1, 5)
                ],
                summary="",
            )
            ctx.rounds_completed = 2
            ctx.coverage_per_round = [{
                "round": 2,
                "results": {
                    f"F{i}": {"coverage": "filter_blocked", "confidence": 0.9,
                              "reasoning": ""}
                    for i in range(1, 5)
                },
            }]
            ctx.continue_decisions = [{
                "round": 2, "decision": "stop_filter_problem",
                "confidence": 0.91,
            }]
            ctx.filter_stats_per_round = [{
                "person_hallucination": {
                    "activated": True, "loss_rate": 1.0,
                    "rejected": 528, "total": 528,
                },
            }]
            ctx.query_anchor = QueryAnchor(type="person",
                                           target="Preis Leistungs",
                                           confidence=0.55)

            node = DiagnosisNode(llm=MagicMock(), max_rounds=3)
            await node.run(ctx)
        finally:
            _restore_adapter(original)

        self.assertEqual(ctx.final_diagnosis["diagnosis"], "filter_too_strict")
        self.assertTrue(ctx.final_diagnosis["is_problematic"])
        self.assertIn("Personen-Filter", ctx.final_diagnosis["remediation"])


# ───────────────────────────────────────────────────────────────────
# FactoidVerificationNode
# ───────────────────────────────────────────────────────────────────


class TestFactoidVerificationNode(unittest.TestCase):

    @async_test
    async def test_skipped_if_no_report(self):
        node = FactoidVerificationNode(llm=MagicMock())
        ctx = _make_ctx()
        ctx.final_report = ""
        applies, reason = node.applies_to(ctx)
        self.assertFalse(applies)

    @async_test
    async def test_full_flow_with_unverified(self):
        llm = MockLLM()
        # extraction
        llm.queue_json([
            {"factoid": "DGX kam 1999", "type": "date",
             "report_position": "..."},
        ])
        # verification
        llm.queue_json([
            {"factoid_index": 1, "verified": "false", "confidence": 0.95,
             "supporting_extract_id": "E0",
             "supporting_quote": "Released 17.10.2024",
             "reasoning": "Widerspruch"},
        ])

        original = _patch_adapter_with(llm)
        try:
            node = FactoidVerificationNode(llm=MagicMock())
            ctx = _make_ctx()
            ctx.final_report = "Die DGX kam 1999 raus."
            ctx.extracts = [
                SourceExtract("u", "t", "F1", "Released 17.10.2024",
                              polarity="positive"),
            ]
            metadata = await node.run(ctx)
        finally:
            _restore_adapter(original)

        self.assertEqual(metadata["unverified"], 1)
        # FactoidVerificationNode does NOT handle high-confidence cases (≥0.9)
        # itself — they go to ReportRevisionNode.
        # The report stays unchanged after FactoidVerificationNode;
        # only ctx.factoid_verifications is set.
        self.assertEqual(ctx.final_report, "Die DGX kam 1999 raus.",
                         "high-confidence case: FactoidVerificationNode "
                         "must not annotate the report itself.")
        self.assertEqual(len(ctx.factoid_verifications), 1)
        self.assertEqual(ctx.factoid_verifications[0]["verified"], "false")
        self.assertEqual(ctx.factoid_verifications[0]["confidence"], 0.95)


# ───────────────────────────────────────────────────────────────────
# Run a complete mini pipeline
# ───────────────────────────────────────────────────────────────────


class TestMultiNodePipeline(unittest.TestCase):

    @async_test
    async def test_full_post_synthesis_pipeline(self):
        """End to end: diagnosis + factoid verification as a DAG."""
        llm = MockLLM()
        # Diagnose
        llm.queue_json({
            "diagnosis": "successful", "confidence": 0.9,
            "remediation": "", "user_message": "ok",
            "reasoning": "Solide Quellenlage.",
        })
        # Extract
        llm.queue_json([
            {"factoid": "Behauptung", "type": "claim",
             "report_position": "..."},
        ])
        # Verify
        llm.queue_json([
            {"factoid_index": 1, "verified": "true", "confidence": 0.85,
             "supporting_extract_id": "E0", "supporting_quote": "X",
             "reasoning": "match"},
        ])

        original = _patch_adapter_with(llm)
        try:
            ctx = _make_ctx()
            ctx.rounds_completed = 1
            ctx.final_report = "## Bericht\n\nBehauptung."
            ctx.extracts = [
                SourceExtract("u", "t", "F1", "X", polarity="positive"),
            ]
            ctx.coverage_per_round = [{
                "round": 1,
                "results": {"F1": {"coverage": "answered", "confidence": 0.9,
                                   "reasoning": ""}},
            }]
            ctx.filter_stats_per_round = []
            ctx.continue_decisions = []
            ctx.query_anchor = QueryAnchor()

            pipeline = PipelineDAG([
                DiagnosisNode(llm=MagicMock(), max_rounds=3),
                FactoidVerificationNode(llm=MagicMock()),
            ])
            await pipeline.run(ctx)
        finally:
            _restore_adapter(original)

        # both nodes ran successfully
        statuses = [r.status for r in ctx.node_results]
        self.assertEqual(statuses, [NodeStatus.OK, NodeStatus.OK])
        self.assertEqual(ctx.final_diagnosis["diagnosis"], "successful")
        self.assertEqual(len(ctx.factoid_verifications), 1)
        self.assertEqual(ctx.factoid_verifications[0]["verified"], "true")


if __name__ == "__main__":
    unittest.main()
