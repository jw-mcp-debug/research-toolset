"""
Smoke test of the DAG path (`run_via_dag`).

Acceptance: `run_via_dag` runs end to end (status='done', a report exists,
the classifiers ran, the diagnosis is set).

Reuses the mock classes from `test_pipeline_smoke.py`.
"""

import json
import sys
import unittest

# Test isolation: see test_pipeline_smoke.py — remove sys.modules
# mutations by other test modules before loading the real modules.
for _mod in (
    "src.connectors.base",
    "src.connectors.rate_limiter",
    "src.config",
    "src.llm.client",
):
    # Only drop stand-ins (they have no __file__). Popping the real module
    # would let a later test module install its own stand-in and leak it
    # into every module imported afterwards.
    if getattr(sys.modules.get(_mod), "__file__", "real") is None:
        sys.modules.pop(_mod, None)

from src.connectors.base import ConnectorRegistry
from src.pipeline.models import SearchResult, SourceType
from tests._helpers import async_test
from tests.test_pipeline_smoke import (
    _MockSearchConnector,
    _MockWebScraper,
    _ScriptedLLM,
)


class TestPipelineSmokeDAG(unittest.TestCase):

    def setUp(self):
        """Protection against mock leaks, as in test_pipeline_smoke.setUp."""
        import sys
        from unittest.mock import MagicMock
        for mod_name in (
            "src.connectors.base",
            "src.connectors.rate_limiter",
            "src.config",
            "src.llm.client",
            "src.pipeline.orchestrator",
        ):
            if mod_name in sys.modules and isinstance(sys.modules[mod_name], MagicMock):
                del sys.modules[mod_name]

    @async_test
    async def test_dag_pipeline_runs_end_to_end(self):
        """Smoke test of the DAG path — the pipeline runs through cleanly."""
        from src.config import PipelineConfig
        from src.pipeline.orchestrator import ResearchOrchestrator

        analysis_response = {
            "summary": "DAG-Smoke-Test",
            "questions": [{
                "id": "F1", "question": "Was ist DGX B300?",
                "search_terms": ["DGX B300"],
                "search_terms_by_lang": {"en": ["DGX B300"]},
                "priority": "hoch",
                "search_langs": ["en"],
                "source_scope": "web",
            }],
            "direct_urls": [], "git_repos": [], "directory_queries": [],
        }

        primary = {
            "You are a research planner": json.dumps(analysis_response),
            "Write the complete report":
                "## DGX B300\n\nEin KI-Beschleuniger.",
            "format agent": json.dumps({"output_schema": {"type": "report"}}),
        }
        harvest = {
            "PERSON or ORGANISATION ANCHOR": json.dumps({
                "anchor_type": "none", "target": "", "confidence": 0.94,
                "reasoning": "Produktanfrage zu Hardware-Spezifikationen ohne Personenbezug.",
            }),
            "search results, assess whether it will contribute": json.dumps([
                {"index": i+1, "relevance": "high", "confidence": 0.85,
                 "reasoning": "Direkter Bezug"}
                for i in range(10)
            ]),
            "Identify all FACTOIDS": json.dumps([
                {"factoid": "DGX B300 ist ein KI-Beschleuniger",
                 "type": "claim", "report_position": "..."},
            ]),
            "For each of the following factoids": json.dumps([
                {"factoid_index": 1, "verified": "true", "confidence": 0.9,
                 "supporting_extract_id": "E0",
                 "supporting_quote": "DGX B300", "reasoning": "match"},
            ]),
            "research question is answered by the": json.dumps({
                "coverage": "answered", "confidence": 0.88,
                "missing_aspects": [], "supporting_extract_ids": ["E0"],
                "reasoning": "Extrakte decken die Frage ab.",
            }),
            "You control an autonomous research run": json.dumps({
                "decision": "stop_done", "confidence": 0.85,
                "next_round_focus": "",
                "reasoning": "Frage ist beantwortet.",
            }),
            "Diagnose the state of this research run": json.dumps({
                "diagnosis": "successful", "confidence": 0.88,
                "remediation": "", "user_message": "ok",
                "reasoning": "Solide Quellenlage.",
            }),
            "extract relevant information from the following source": (
                "[F1] [POSITIV] Fakt: DGX B300 ist ein KI-Beschleuniger\n"
                "     Verlässlichkeit: hoch\n"
            ),
        }

        llm = _ScriptedLLM(primary_responses=primary, harvest_responses=harvest)

        registry = ConnectorRegistry()
        search_results = [SearchResult(
            title="NVIDIA DGX B300",
            url="https://nvidia.com/dgx-b300",
            snippet="DGX B300 is a flagship AI accelerator with 14kW TDP.",
            source_type=SourceType.WEB_SEARCH,
        )]
        registry.register(_MockSearchConnector(search_results),
                          is_search_engine=True)
        registry.register(_MockWebScraper({}), is_web_scraper=True)

        config = PipelineConfig()
        config.max_rounds = 1
        config.data_dir = "/tmp/recherche_dag_smoke"

        orch = ResearchOrchestrator(
            llm=llm, connectors=registry, config=config,
        )

        progress_events = []
        async def progress(event_type, data=None):
            progress_events.append((event_type, data))

        # ── Run the DAG path ──
        ctx = await orch.run_via_dag(
            query="Was ist DGX B300?",
            chat_history=[],
            context_docs="",
            template_name="",
            progress_callback=progress,
            mode="web",
        )

        # ── Acceptance ──
        self.assertEqual(
            ctx.status, "done",
            f"wrong status: {ctx.status} — "
            f"error_message: {getattr(ctx, 'error_message', '?')}",
        )
        self.assertIsNotNone(ctx.research_plan)
        self.assertGreater(len(ctx.sources), 0, "no sources!")
        self.assertGreater(len(ctx.extracts), 0, "no extracts!")

        # the classifier path ran
        self.assertGreater(len(ctx.coverage_per_round), 0)
        self.assertIsNotNone(ctx.final_diagnosis)

        # report exists
        self.assertGreater(len(ctx.final_report), 20)

        # node_results is filled — the DAG engine logged
        self.assertGreater(len(ctx.node_results), 0)

        # the factoid verification ran
        self.assertGreaterEqual(len(ctx.factoid_verifications), 1)

        # the engine fired at least one "node_done" event
        node_events = [e for e in progress_events if e[0] in (
            "node_start", "node_done", "node_failed",
        )]
        self.assertGreater(
            len(node_events), 5,
            "expected several DAG node events",
        )

    @async_test
    async def test_dag_pipeline_with_no_new_sources(self):
        """Robustness: no search results → the loop stops, the pipeline ends cleanly."""
        from src.config import PipelineConfig
        from src.pipeline.orchestrator import ResearchOrchestrator

        primary = {
            "You are a research planner": json.dumps({
                "summary": "X",
                "questions": [{
                    "id": "F1", "question": "?",
                    "search_terms": ["x"], "priority": "hoch",
                }],
                "direct_urls": [], "git_repos": [], "directory_queries": [],
            }),
            "Write the complete report": "Mini-Bericht ohne Quellen",
        }
        harvest = {
            "PERSON or ORGANISATION ANCHOR": json.dumps({
                "anchor_type": "none", "target": "", "confidence": 0.9,
                "reasoning": "ohne klaren Bezug",
            }),
            "Diagnose the state of this research run": json.dumps({
                "diagnosis": "data_scarcity", "confidence": 0.85,
                "user_message": "Keine Quellen gefunden",
                "remediation": "", "reasoning": "leer.",
            }),
            "Identify all FACTOIDS": "[]",
        }

        llm = _ScriptedLLM(primary, harvest)

        registry = ConnectorRegistry()
        registry.register(_MockSearchConnector([]),  # NO results
                          is_search_engine=True)
        registry.register(_MockWebScraper({}), is_web_scraper=True)

        config = PipelineConfig()
        config.max_rounds = 1
        config.data_dir = "/tmp/recherche_dag_smoke_empty"

        orch = ResearchOrchestrator(
            llm=llm, connectors=registry, config=config,
        )

        async def progress(*a, **kw): pass

        ctx = await orch.run_via_dag(
            query="x", chat_history=[],
            context_docs="", template_name="",
            progress_callback=progress, mode="web",
        )

        # the pipeline ends cleanly, without a crash
        self.assertIn(ctx.status, ("done", "error"))
        # a report is produced anyway (even without sources)
        self.assertIsNotNone(ctx.final_report)


if __name__ == "__main__":
    unittest.main()
