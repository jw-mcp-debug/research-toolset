"""
Tests for the routing of `run()` to the DAG paths.

Acceptance:
  - `run()` with standard arguments delegates to `run_via_dag()`
    (= ctx.node_results is filled).
  - `run()` with `plan_only=True` goes through `run_via_dag_plan_only()`.
  - `run()` with `existing_ctx + skip_decompose` goes through the DAG path;
    FormatAgent/Analysis are skipped.
  - a crash in the DAG path propagates (no silent fallback).
"""

import json
import unittest

from src.connectors.base import ConnectorRegistry
from src.pipeline.models import (
    HarvestContext,
    OutputSchema,
    ResearchPlan,
    ResearchQuestion,
    SearchResult,
    SourceType,
)
from tests._helpers import async_test
from tests.test_pipeline_smoke import (
    _MockSearchConnector,
    _MockWebScraper,
    _ScriptedLLM,
)


def _build_orchestrator():
    """Standard setup: LLM + connectors with mock answers."""
    primary = {
        "Beantwortet folgende Anfrage": json.dumps({
            "summary": "x", "questions": [{
                "id": "F1", "question": "?",
                "search_terms": ["x"], "priority": "hoch",
            }], "direct_urls": [], "git_repos": [], "directory_queries": [],
        }),
        "Erstelle einen strukturierten Bericht": "Bericht.",
    }
    harvest = {
        "PERSONEN- oder ORGANISATIONS-ANKER": json.dumps({
            "anchor_type": "none", "target": "", "confidence": 0.9,
            "reasoning": "Produktanfrage ohne Personenbezug.",
        }),
        "Suchergebnisse, ob es zur Beantwortung": json.dumps([
            {"index": i+1, "relevance": "high",
             "confidence": 0.9, "reasoning": "ja"}
            for i in range(10)
        ]),
        "Identifiziere alle FAKTOIDE": "[]",
        "Recherche-Frage durch die vorliegenden Extrakte": json.dumps({
            "coverage": "answered", "confidence": 0.88,
            "missing_aspects": [], "reasoning": "ok.",
        }),
        "Du steuerst eine autonome Recherche": json.dumps({
            "decision": "stop_done", "confidence": 0.85,
            "reasoning": "fertig.",
        }),
        "Diagnostiziere den Zustand": json.dumps({
            "diagnosis": "successful", "confidence": 0.88,
            "user_message": "ok", "remediation": "",
            "reasoning": "ok.",
        }),
        "Extrahiere relevante Informationen": (
            "[F1] [POSITIV] Fakt: x\n     Verlässlichkeit: hoch"
        ),
    }
    llm = _ScriptedLLM(primary, harvest)
    reg = ConnectorRegistry()
    reg.register(_MockSearchConnector([
        SearchResult(
            "Test", "https://test.com/1",
            "snippet long enough for filter passage",
            SourceType.WEB_SEARCH,
        ),
    ]), is_search_engine=True)
    reg.register(_MockWebScraper({}), is_web_scraper=True)

    from src.config import PipelineConfig
    from src.pipeline.orchestrator import ResearchOrchestrator
    config = PipelineConfig()
    config.max_rounds = 1
    config.data_dir = "/tmp/rh_switchover_test"
    return ResearchOrchestrator(llm=llm, connectors=reg, config=config)


async def _noop_cb(*args, **kwargs):
    pass


class TestSwitchover(unittest.TestCase):

    @async_test
    async def test_standard_run_delegates_to_dag(self):
        """run() with standard arguments → DAG path."""
        orch = _build_orchestrator()
        ctx = await orch.run(
            query="x", chat_history=[],
            context_docs="", template_name="",
            progress_callback=_noop_cb,
            mode="web",
        )

        self.assertEqual(ctx.status, "done")
        # node_results is specific to the DAG path
        self.assertGreater(
            len(ctx.node_results), 0,
            "expected filled node_results — did the DAG path not run?",
        )

    @async_test
    async def test_plan_only_uses_dag(self):
        """run() with plan_only=True → DAG plan-only path."""
        orch = _build_orchestrator()
        ctx = await orch.run(
            query="x", chat_history=[],
            context_docs="", template_name="",
            progress_callback=_noop_cb,
            mode="web",
            plan_only=True,
        )

        # Plan-Preview-Status
        self.assertEqual(ctx.status, "plan_ready")
        # DAG path: node_results filled (with FormatAgent + Analysis)
        self.assertGreater(
            len(getattr(ctx, "node_results", [])), 0,
            "plan_only should go through the DAG plan-only path",
        )
        # NOT 13+ nodes as in the full pipeline — only the plan phase
        self.assertLessEqual(
            len(ctx.node_results), 5,
            f"plan_only should run only plan nodes, "
            f"but ran {len(ctx.node_results)}",
        )

    @async_test
    async def test_existing_ctx_uses_dag_skips_plan_phase(self):
        """run() with existing_ctx + skip_decompose → DAG path,
        FormatAgent/Analysis are skipped."""
        orch = _build_orchestrator()
        # Create a prepared context (as after plan_only)
        existing = HarvestContext(query="x", chat_history=[])
        existing.output_schema = OutputSchema(
            title="X", format_type="report", sections=["A"],
        )
        existing.research_plan = ResearchPlan(
            summary="schon da mit Text",
            questions=[ResearchQuestion(
                id="F1", question="?",
                search_terms=["x"], priority="hoch",
            )],
        )

        ctx = await orch.run(
            query="x", chat_history=[],
            context_docs="", template_name="",
            progress_callback=_noop_cb,
            mode="web",
            existing_ctx=existing,
            skip_decompose=True,
        )

        # Resume mode goes through the DAG → node_results filled
        self.assertGreater(
            len(getattr(ctx, "node_results", [])), 0,
            "resume mode should go through the DAG path",
        )
        # FormatAgent and Analysis should appear as SKIPPED
        from src.pipeline.dag import NodeStatus
        skipped_names = [
            r.node_name for r in ctx.node_results
            if r.status == NodeStatus.SKIPPED
        ]
        self.assertIn(
            "format_agent", skipped_names,
            f"format_agent expected skipped, found: {skipped_names}",
        )
        self.assertIn(
            "analysis", skipped_names,
            f"analysis expected skipped, found: {skipped_names}",
        )

    @async_test
    async def test_dag_failure_propagates(self):
        """A crash in the DAG path propagates — no silent fallback.

        No fallback is wanted: a crash in the DAG path must become
        visible as a bug, not be covered up by faulty masking.
        """
        orch = _build_orchestrator()
        # Replace run_via_dag by a broken version
        async def broken(**kw):
            raise RuntimeError("DAG broken")
        orch.run_via_dag = broken  # type: ignore

        with self.assertRaises(RuntimeError) as raised:
            await orch.run(
                query="x", chat_history=[],
                context_docs="", template_name="",
                progress_callback=_noop_cb,
                mode="web",
            )
        self.assertIn("DAG broken", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
