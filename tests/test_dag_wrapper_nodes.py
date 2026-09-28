"""
Tests for the wrapper nodes of the DAG pipeline.

These tests do NOT check the logic of the wrapped methods — that is covered
by the classifier tests and the smoke test. They check the wrapper
properties:

  - applies_to requires the right ctx preconditions
  - the wrapped method is called correctly
  - the result ends up in the right ctx field
  - metadata is returned sensibly
  - fail_mode is as specified (STOP for structural nodes, CONTINUE for
    optional ones)

Strategy: a MagicMock orchestrator whose methods return given values. We
verify that the node calls the method and processes its result correctly.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from src.pipeline.dag import FailMode, NodeStatus, PipelineDAG
from src.pipeline.dag_nodes import (
    AnalysisNode,
    ContradictionCheckNode,
    DiagnosisBannerNode,
    FormatAgentNode,
    HarvestNode,
    SearchAndFetchNode,
    SynthesisNode,
)
from src.pipeline.models import (
    HarvestContext,
    HarvestResult,
    OutputSchema,
    ResearchPlan,
    ResearchQuestion,
    SourceDocument,
    SourceExtract,
    SourceType,
)
from tests._helpers import async_test


def _make_ctx(**fields) -> HarvestContext:
    ctx = HarvestContext(query=fields.pop("query", "Test"))
    for k, v in fields.items():
        setattr(ctx, k, v)
    return ctx


# ───────────────────────────────────────────────────────────────────
# FormatAgentNode
# ───────────────────────────────────────────────────────────────────


class TestFormatAgentNode(unittest.TestCase):

    def test_fail_mode_is_stop(self):
        """Structurally important: no schema, no synthesis."""
        self.assertEqual(FormatAgentNode.fail_mode, FailMode.STOP)

    @async_test
    async def test_skipped_if_schema_already_set(self):
        ctx = _make_ctx()
        ctx.output_schema = OutputSchema()
        node = FormatAgentNode(orchestrator=MagicMock())

        applies, reason = node.applies_to(ctx)
        self.assertFalse(applies)
        self.assertIn("already set", reason)

    @async_test
    async def test_calls_orchestrator_method(self):
        orch = MagicMock()
        expected = OutputSchema(title="Mock-Schema",
                                format_type="report",
                                sections=["A", "B"])
        orch._run_format_agent = AsyncMock(return_value=expected)

        ctx = _make_ctx()
        ctx.chat_history = [{"role": "user", "content": "hi"}]

        node = FormatAgentNode(
            orchestrator=orch,
            context_docs="docs",
            template_name="general",
        )
        meta = await node.run(ctx)

        # the method was called with the right arguments
        orch._run_format_agent.assert_called_once_with(
            query="Test",
            chat_history=[{"role": "user", "content": "hi"}],
            context_docs="docs",
            template_name="general",
        )
        # result in ctx
        self.assertIs(ctx.output_schema, expected)
        # Metadata
        self.assertEqual(meta["format"], "report")
        self.assertEqual(meta["sections"], 2)


# ───────────────────────────────────────────────────────────────────
# AnalysisNode
# ───────────────────────────────────────────────────────────────────


class TestAnalysisNode(unittest.TestCase):

    def test_fail_mode_is_stop(self):
        """No plan, no research."""
        self.assertEqual(AnalysisNode.fail_mode, FailMode.STOP)

    @async_test
    async def test_skipped_without_schema(self):
        ctx = _make_ctx()
        # output_schema NOT set
        node = AnalysisNode(orchestrator=MagicMock())

        applies, reason = node.applies_to(ctx)
        self.assertFalse(applies)
        self.assertIn("output_schema", reason)

    @async_test
    async def test_skipped_if_plan_already_set(self):
        """Plan preview gate scenario: the plan exists already, not again."""
        ctx = _make_ctx()
        ctx.output_schema = OutputSchema()
        ctx.research_plan = ResearchPlan(summary="schon da", questions=[])

        node = AnalysisNode(orchestrator=MagicMock())
        applies, _ = node.applies_to(ctx)
        self.assertFalse(applies)

    @async_test
    async def test_calls_method_and_writes_plan(self):
        plan = ResearchPlan(
            summary="Test-Plan-Zusammenfassung mit ausreichend Text.",
            questions=[ResearchQuestion(
                id="F1", question="?",
                search_terms=["x"], priority="hoch",
            )],
        )
        orch = MagicMock()
        orch._run_analysis = AsyncMock(return_value=plan)

        ctx = _make_ctx(output_schema=OutputSchema())
        node = AnalysisNode(orchestrator=orch, context_docs="ctx")
        meta = await node.run(ctx)

        orch._run_analysis.assert_called_once()
        self.assertIs(ctx.research_plan, plan)
        self.assertEqual(meta["questions"], 1)


# ───────────────────────────────────────────────────────────────────
# SearchAndFetchNode
# ───────────────────────────────────────────────────────────────────


class TestSearchAndFetchNode(unittest.TestCase):

    @async_test
    async def test_skipped_without_plan(self):
        ctx = _make_ctx()
        node = SearchAndFetchNode(
            orchestrator=MagicMock(), round_num=0,
            progress_callback=AsyncMock(),
        )
        applies, _ = node.applies_to(ctx)
        self.assertFalse(applies)

    @async_test
    async def test_appends_new_sources(self):
        new = [
            SourceDocument(SourceType.WEB_PAGE, "https://a.com", "A", "x"),
            SourceDocument(SourceType.WEB_PAGE, "https://b.com", "B", "y"),
        ]
        orch = MagicMock()
        orch._run_search_and_fetch = AsyncMock(return_value=new)

        ctx = _make_ctx()
        ctx.research_plan = ResearchPlan(summary="", questions=[])
        # ctx has no sources yet
        self.assertEqual(ctx.sources, [])

        node = SearchAndFetchNode(
            orchestrator=orch, round_num=0,
            progress_callback=AsyncMock(),
        )
        meta = await node.run(ctx)

        self.assertEqual(len(ctx.sources), 2)
        self.assertEqual(meta["new_sources"], 2)


# ───────────────────────────────────────────────────────────────────
# HarvestNode
# ───────────────────────────────────────────────────────────────────


class TestHarvestNode(unittest.TestCase):

    @async_test
    async def test_skipped_without_new_sources(self):
        node = HarvestNode(
            orchestrator=MagicMock(),
            new_sources=[],
            progress_callback=AsyncMock(),
        )
        ctx = _make_ctx()
        ctx.research_plan = ResearchPlan(summary="", questions=[])

        applies, reason = node.applies_to(ctx)
        self.assertFalse(applies)
        self.assertIn("no new sources", reason)

    @async_test
    async def test_only_positive_extracts_added(self):
        """Only polarity='positive' goes into ctx.extracts."""
        new_src = SourceDocument(SourceType.WEB_PAGE, "u", "t", "c")
        harvest_results = [
            HarvestResult(
                source_url="u", source_title="t",
                extracts=[
                    SourceExtract("u", "t", "F1", "positiv-1",
                                  polarity="positive"),
                    SourceExtract("u", "t", "F1", "negativ",
                                  polarity="negative"),
                    SourceExtract("u", "t", "F1", "meta",
                                  polarity="meta"),
                    SourceExtract("u", "t", "F1", "positiv-2",
                                  polarity="positive"),
                ],
                is_relevant=True,
            ),
        ]

        orch = MagicMock()
        orch._run_harvest = AsyncMock(return_value=harvest_results)
        orch._filter_stats_per_round = [{"x": {"y": 1}}]

        ctx = _make_ctx()
        ctx.research_plan = ResearchPlan(summary="", questions=[])

        node = HarvestNode(
            orchestrator=orch, new_sources=[new_src],
            progress_callback=AsyncMock(),
        )
        meta = await node.run(ctx)

        # only 2 positive extracts
        self.assertEqual(len(ctx.extracts), 2)
        self.assertEqual(meta["new_extracts"], 2)
        # harvest_results are in it
        self.assertEqual(len(ctx.harvest_results), 1)
        # filter_stats were mirrored
        self.assertEqual(ctx.filter_stats_per_round, [{"x": {"y": 1}}])


# ───────────────────────────────────────────────────────────────────
# Consistency regression: classifier visibility + anchor cache
# ───────────────────────────────────────────────────────────────────


class TestClassifierCallMirroring(unittest.TestCase):
    """SearchAndFetchNode must mirror classifier calls made inside the
    orchestrator (search_scope, off-topic filter) into ctx.classifier_calls
    — otherwise they are invisible in the pipeline-run tab. Delta
    mirroring: no duplicates across rounds."""

    @async_test
    async def test_orch_calls_mirrored_to_ctx(self):
        orch = MagicMock()
        orch._run_search_and_fetch = AsyncMock(return_value=[])
        orch._classifier_calls = ["search_scope_call", "offtopic_r0"]
        orch._cc_mirrored = 0

        ctx = _make_ctx()
        ctx.research_plan = ResearchPlan(summary="", questions=[])

        node = SearchAndFetchNode(
            orchestrator=orch, round_num=0,
            progress_callback=AsyncMock(),
        )
        await node.run(ctx)

        self.assertEqual(
            ctx.classifier_calls,
            ["search_scope_call", "offtopic_r0"],
        )
        self.assertEqual(orch._cc_mirrored, 2)

    @async_test
    async def test_no_duplicate_across_rounds(self):
        orch = MagicMock()
        orch._run_search_and_fetch = AsyncMock(return_value=[])
        orch._classifier_calls = ["scope", "ot_r0"]
        orch._cc_mirrored = 0
        ctx = _make_ctx()
        ctx.research_plan = ResearchPlan(summary="", questions=[])

        n0 = SearchAndFetchNode(orch, 0, AsyncMock())
        await n0.run(ctx)
        # round 1: off-topic appends new calls
        orch._classifier_calls = ["scope", "ot_r0", "ot_r1"]
        n1 = SearchAndFetchNode(orch, 1, AsyncMock())
        await n1.run(ctx)

        # every call exactly ONCE in ctx
        self.assertEqual(
            ctx.classifier_calls, ["scope", "ot_r0", "ot_r1"]
        )

    @async_test
    async def test_mirror_failure_never_breaks_pipeline(self):
        orch = MagicMock()
        orch._run_search_and_fetch = AsyncMock(return_value=[])
        # _classifier_calls not iterable → the mirror block must swallow
        # the exception, run() succeeds anyway
        orch._classifier_calls = None
        orch._cc_mirrored = 0
        ctx = _make_ctx()
        ctx.research_plan = ResearchPlan(summary="", questions=[])

        node = SearchAndFetchNode(orch, 0, AsyncMock())
        meta = await node.run(ctx)  # must NOT raise
        self.assertEqual(meta["new_sources"], 0)


class TestQueryAnchorCacheBridge(unittest.TestCase):
    """HarvestNode must bridge ctx.query_anchor into the orchestrator cache
    (_query_anchor), so that _run_harvest does NOT classify the anchor a
    second time."""

    @async_test
    async def test_bridges_ctx_anchor_to_orch_cache(self):
        orch = MagicMock()
        orch._run_harvest = AsyncMock(return_value=[])
        orch._filter_stats_per_round = []
        orch._query_anchor = None  # cache empty (QueryAnchorNode does not set it)

        ctx = _make_ctx()
        ctx.research_plan = ResearchPlan(summary="", questions=[])
        ctx.query_anchor = "ANCHOR_FROM_NODE"

        node = HarvestNode(
            orchestrator=orch,
            new_sources=[SourceDocument(SourceType.WEB_PAGE, "u", "t", "c")],
            progress_callback=AsyncMock(),
        )
        await node.run(ctx)

        # bridge built → _run_harvest sees the cache and does not classify
        # again
        self.assertEqual(orch._query_anchor, "ANCHOR_FROM_NODE")

    @async_test
    async def test_does_not_overwrite_existing_cache(self):
        orch = MagicMock()
        orch._run_harvest = AsyncMock(return_value=[])
        orch._filter_stats_per_round = []
        orch._query_anchor = "ALREADY_CACHED"

        ctx = _make_ctx()
        ctx.research_plan = ResearchPlan(summary="", questions=[])
        ctx.query_anchor = "DIFFERENT"

        node = HarvestNode(
            orchestrator=orch,
            new_sources=[SourceDocument(SourceType.WEB_PAGE, "u", "t", "c")],
            progress_callback=AsyncMock(),
        )
        await node.run(ctx)

        # an existing cache is NOT overwritten
        self.assertEqual(orch._query_anchor, "ALREADY_CACHED")


# ───────────────────────────────────────────────────────────────────
# ContradictionCheckNode
# ───────────────────────────────────────────────────────────────────


class TestContradictionCheckNode(unittest.TestCase):

    @async_test
    async def test_skipped_when_too_few_extracts(self):
        ctx = _make_ctx()
        ctx.extracts = [
            SourceExtract("u", "t", "F1", "x") for _ in range(2)
        ]
        node = ContradictionCheckNode(
            orchestrator=MagicMock(), min_extracts=4,
        )
        applies, reason = node.applies_to(ctx)
        self.assertFalse(applies)

    @async_test
    async def test_writes_contradictions_to_ctx(self):
        contradictions = [
            {"question_id": "F1", "summary": "A widerspricht B"},
        ]
        orch = MagicMock()
        orch._check_contradictions = AsyncMock(return_value=contradictions)

        ctx = _make_ctx()
        ctx.extracts = [
            SourceExtract("u", "t", "F1", f"x{i}") for i in range(5)
        ]
        node = ContradictionCheckNode(orchestrator=orch, min_extracts=4)

        await node.run(ctx)
        self.assertEqual(ctx.contradiction_warnings, contradictions)


# ───────────────────────────────────────────────────────────────────
# SynthesisNode
# ───────────────────────────────────────────────────────────────────


class TestSynthesisNode(unittest.TestCase):

    def test_fail_mode_is_stop(self):
        """Without a report the pipeline has no result."""
        self.assertEqual(SynthesisNode.fail_mode, FailMode.STOP)

    @async_test
    async def test_skipped_without_plan(self):
        ctx = _make_ctx()
        node = SynthesisNode(
            orchestrator=MagicMock(),
            context_docs="",
            progress_callback=AsyncMock(),
        )
        applies, _ = node.applies_to(ctx)
        self.assertFalse(applies)

    @async_test
    async def test_writes_report_to_ctx(self):
        orch = MagicMock()
        orch._run_synthesis = AsyncMock(return_value="## Mock-Bericht\n\nInhalt")

        ctx = _make_ctx()
        ctx.research_plan = ResearchPlan(summary="", questions=[])

        node = SynthesisNode(
            orchestrator=orch,
            context_docs="docs",
            progress_callback=AsyncMock(),
        )
        meta = await node.run(ctx)

        self.assertIn("Mock-Bericht", ctx.final_report)
        self.assertGreater(meta["report_length"], 0)


# ───────────────────────────────────────────────────────────────────
# DiagnosisBannerNode
# ───────────────────────────────────────────────────────────────────


class TestDiagnosisBannerNode(unittest.TestCase):

    @async_test
    async def test_skipped_when_no_diagnosis(self):
        ctx = _make_ctx()
        ctx.final_report = "## Bericht"
        # final_diagnosis is None by default

        node = DiagnosisBannerNode()
        applies, reason = node.applies_to(ctx)
        self.assertFalse(applies)
        self.assertIn("diagnosis", reason)

    @async_test
    async def test_skipped_when_diagnosis_successful(self):
        ctx = _make_ctx()
        ctx.final_report = "## Bericht"
        ctx.final_diagnosis = {
            "diagnosis": "successful", "is_problematic": False,
            "is_successful": True, "user_message": "ok", "remediation": "",
        }
        node = DiagnosisBannerNode()
        applies, _ = node.applies_to(ctx)
        self.assertFalse(applies)

    @async_test
    async def test_skipped_when_no_report(self):
        ctx = _make_ctx()
        ctx.final_report = ""
        ctx.final_diagnosis = {"is_problematic": True, "diagnosis": "x"}
        node = DiagnosisBannerNode()
        applies, _ = node.applies_to(ctx)
        self.assertFalse(applies)

    @async_test
    async def test_appends_banner(self):
        """CRITICAL: the diagnosis banner is appended."""
        ctx = _make_ctx()
        ctx.final_report = "## Bericht\n\nKurz."
        ctx.final_diagnosis = {
            "diagnosis": "filter_too_strict",
            "is_problematic": True,
            "is_successful": False,
            "user_message": "Filter hat alle Extrakte verworfen.",
            "remediation": "Recherche ohne Filter neu starten.",
        }

        node = DiagnosisBannerNode()
        await node.run(ctx)

        self.assertIn("⚠️ Note on the research run", ctx.final_report)
        self.assertIn("filter_too_strict", ctx.final_report)
        self.assertIn("Filter hat alle Extrakte verworfen", ctx.final_report)
        self.assertIn("**Recommendation:**", ctx.final_report)


# ───────────────────────────────────────────────────────────────────
# Mini pipeline: structural nodes working together
# ───────────────────────────────────────────────────────────────────


class TestStructuralPipeline(unittest.TestCase):

    @async_test
    async def test_format_agent_then_analysis_then_synthesis(self):
        """Mini pipeline: schema → plan → report.

        Verifies that the wrappers can be chained — the output of one node
        is the input of the next.
        """
        # mock orchestrator with the three methods
        orch = MagicMock()
        orch._run_format_agent = AsyncMock(
            return_value=OutputSchema(title="X", format_type="report",
                                       sections=["A"])
        )
        orch._run_analysis = AsyncMock(
            return_value=ResearchPlan(
                summary="Plan-Beschreibung mit ausreichend Text.",
                questions=[ResearchQuestion(
                    id="F1", question="?",
                    search_terms=["x"], priority="hoch",
                )],
            )
        )
        orch._run_synthesis = AsyncMock(return_value="## Mein Bericht")

        ctx = _make_ctx(query="Was ist X?")
        progress = AsyncMock()

        pipeline = PipelineDAG([
            FormatAgentNode(orchestrator=orch),
            AnalysisNode(orchestrator=orch),
            SynthesisNode(
                orchestrator=orch,
                context_docs="",
                progress_callback=progress,
            ),
        ])
        await pipeline.run(ctx, progress_callback=progress)

        # all three nodes successful
        statuses = [r.status for r in ctx.node_results]
        self.assertEqual(statuses, [NodeStatus.OK] * 3)
        # final product
        self.assertIn("Mein Bericht", ctx.final_report)
        # methods called
        orch._run_format_agent.assert_called_once()
        orch._run_analysis.assert_called_once()
        orch._run_synthesis.assert_called_once()

    @async_test
    async def test_format_agent_failure_aborts_pipeline(self):
        """fail_mode=STOP: a format-agent error stops the pipeline."""
        orch = MagicMock()
        orch._run_format_agent = AsyncMock(
            side_effect=RuntimeError("Format-Agent kaputt"),
        )
        orch._run_analysis = AsyncMock()  # must not be called

        ctx = _make_ctx()
        pipeline = PipelineDAG([
            FormatAgentNode(orchestrator=orch),
            AnalysisNode(orchestrator=orch),
        ])

        with self.assertRaises(RuntimeError) as raised:
            await pipeline.run(ctx)

        self.assertIn("Format-Agent kaputt", str(raised.exception))
        # the analysis node did NOT run
        orch._run_analysis.assert_not_called()


if __name__ == "__main__":
    unittest.main()
