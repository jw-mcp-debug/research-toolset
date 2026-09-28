"""
Pipeline smoke test — an end-to-end safety net.

This test runs a complete research from start to finish — with minimal
mocks for DualLLMClient and ConnectorRegistry. It is NOT a correctness
test of individual components (the unit tests cover those) but a smoke
test: "does the pipeline machinery hold together in an end-to-end run?"

Acceptance:
  - ctx.status == "done"
  - ctx.research_plan is set with ≥ 1 question
  - ctx.sources, ctx.extracts are filled
  - the classifier path ran (ctx.coverage_per_round, ctx.continue_decisions)
  - ctx.final_diagnosis is set
  - ctx.final_report is not empty
"""

import asyncio
import json
import sys
import unittest

# Test isolation: other test modules (e.g. test_orchestrator_plan_gate)
# patch src.connectors.base in sys.modules. That stays there and we
# would get a mock module instead of the real one. Before the import we
# remove potentially contaminated cache entries, so that the real code
# is loaded.
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

from src.connectors.base import ConnectorRegistry, BaseConnector
from src.pipeline.models import (
    SearchResult,
    SourceDocument,
    SourceType,
)
from tests._helpers import async_test


# ───────────────────────────────────────────────────────────────────
# Mock DualLLMClient (adapts to calls dynamically)
# ───────────────────────────────────────────────────────────────────


class _ScriptedLLM:
    """Mock LLM with scripted answers plus a default heuristic."""

    def __init__(self, primary_responses: dict, harvest_responses: dict):
        # Mapping prompt keyword → answer. The first match is taken.
        self.primary_responses = primary_responses
        self.harvest_responses = harvest_responses
        self.harvest_semaphore = asyncio.Semaphore(10)
        self._call_log: list = []

    def _match(self, table: dict, content: str) -> str | None:
        for keyword, response in table.items():
            if keyword.lower() in content.lower():
                return response
        return None

    async def primary_complete_json(
        self, messages: list[dict], max_tokens: int | None = None,
    ) -> dict:
        body = "\n".join(m.get("content", "") for m in messages)
        self._call_log.append(("primary_json", body[:200]))
        match = self._match(self.primary_responses, body)
        if match is not None:
            return json.loads(match) if isinstance(match, str) else match
        # Default: an empty dict is enough for the callers. They have defaults.
        return {}

    async def primary_complete(
        self, messages: list[dict], max_tokens: int | None = None, **kwargs,
    ) -> str:
        """Plain completion (map phase, summary): matched text or a default."""
        body = "\n".join(m.get("content", "") for m in messages)
        self._call_log.append(("primary", body[:200]))
        match = self._match(self.primary_responses, body)
        return match if isinstance(match, str) else "Mock answer to the question."

    async def primary_stream(self, messages, max_tokens=None):
        body = "\n".join(m.get("content", "") for m in messages)
        self._call_log.append(("primary_stream", body[:200]))
        # answer as a single chunk
        match = self._match(self.primary_responses, body)
        text = match if isinstance(match, str) else "Mock-Bericht über das Thema."
        for chunk in text.split(" "):
            yield chunk + " "

    async def harvest_complete(
        self, messages: list[dict], max_tokens: int | None = None,
    ) -> str:
        async with self.harvest_semaphore:
            body = "\n".join(m.get("content", "") for m in messages)
            self._call_log.append(("harvest", body[:200]))
            match = self._match(self.harvest_responses, body)
            if match is not None:
                return match
            return ""

    def get_usage_stats(self) -> dict:
        return {
            "primary": {"model": "mock", "requests": 0, "prompt_tokens": 0,
                        "completion_tokens": 0, "total_tokens": 0},
            "harvest": {"model": "mock", "requests": 0, "prompt_tokens": 0,
                        "completion_tokens": 0, "total_tokens": 0},
        }

    def stop(self):
        pass


# ───────────────────────────────────────────────────────────────────
# Mock-Connector + Registry
# ───────────────────────────────────────────────────────────────────


class _MockSearchConnector(BaseConnector):
    name = "mock_search"

    def __init__(self, search_results: list[SearchResult]):
        self._search_results = search_results

    async def search(self, query, max_results=10, **kwargs):
        return self._search_results[:max_results]

    async def fetch(self, url):
        # the search connector does not fetch normally
        return SourceDocument(SourceType.WEB_PAGE, url, "Mock", "Mock-Inhalt", {})

    def can_handle(self, url):
        return False


class _MockWebScraper(BaseConnector):
    name = "mock_scraper"

    def __init__(self, url_to_doc: dict[str, SourceDocument]):
        self._url_to_doc = url_to_doc

    async def search(self, query, max_results=10, **kwargs):
        return []

    async def fetch(self, url):
        if url in self._url_to_doc:
            return self._url_to_doc[url]
        # Default: minimal document, so that the listing filter lets it through
        return SourceDocument(
            SourceType.WEB_PAGE, url,
            f"Mock {url}",
            "Mock-Inhalt für die Smoke-Test-Pipeline. " * 20,
            {},
        )

    def can_handle(self, url):
        return True


# ───────────────────────────────────────────────────────────────────
# Smoke test
# ───────────────────────────────────────────────────────────────────


class TestPipelineSmokeRun(unittest.TestCase):

    def setUp(self):
        """Protection against mock leaks: other test modules
        (test_orchestrator_plan_gate etc.) install MagicMock modules in
        sys.modules. We clean them up before every test, so that the real
        code is loaded."""
        import sys
        for mod_name in (
            "src.connectors.base",
            "src.connectors.rate_limiter",
            "src.config",
            "src.llm.client",
            "src.pipeline.orchestrator",
        ):
            if mod_name in sys.modules:
                # only remove it if it is a MagicMock
                from unittest.mock import MagicMock
                if isinstance(sys.modules[mod_name], MagicMock):
                    del sys.modules[mod_name]

    @async_test
    async def test_pipeline_runs_end_to_end(self):
        from src.config import PipelineConfig
        from src.pipeline.orchestrator import ResearchOrchestrator

        # ── 1. LLM scripts ──
        # Format agent: skip for now (result {} → defaults)
        # Analysis phase: must deliver ResearchPlan JSON
        analysis_response = {
            "summary": "Smoke-Test-Recherche",
            "questions": [
                {
                    "id": "F1",
                    "question": "Was ist DGX B300?",
                    "search_terms": ["DGX B300"],
                    "search_terms_by_lang": {"en": ["DGX B300"]},
                    "priority": "hoch",
                    "search_langs": ["en"],
                    "source_scope": "web",
                },
            ],
            "direct_urls": [],
            "git_repos": [],
            "directory_queries": [],
        }

        # Synthesis: a mock report
        synthesis_response = (
            "## Bericht über DGX B300\n\n"
            "Die DGX B300 ist ein NVIDIA-Server. Sie verbraucht typischerweise 14kW.\n"
        )

        # Classifier answers (on the harvest LLM)
        # We supply the classifier calls via keyword matching.
        # The order is not guaranteed, so register them all in parallel.

        primary = {
            "You are a research planner": json.dumps(analysis_response),
            "Write the complete report": synthesis_response,
            # Format agent (returns direct mode)
            "format agent": json.dumps({"output_schema": {"type": "report"}}),
        }
        harvest = {
            # Query anchor — unique prompt start
            "PERSON or ORGANISATION ANCHOR": json.dumps({
                "anchor_type": "none",
                "target": "",
                "confidence": 0.94,
                "reasoning": "Produktanfrage zu Hardware-Spezifikationen ohne Personenbezug.",
            }),
            # judge_source_relevance (batched) — NB: must come BEFORE the
            # coverage match, because "Bewerte" at the start would conflict as a
            # prefix; specific keyword:
            "search results, assess whether it will contribute": json.dumps([
                {"index": i+1, "relevance": "high", "confidence": 0.85,
                 "reasoning": "Direkter Bezug zum Produkt"}
                for i in range(10)
            ]),
            # extract_factoids — first! otherwise "Faktoide" matches for verify
            "Identify all FACTOIDS": json.dumps([
                {"factoid": "DGX B300 ist ein KI-Beschleuniger",
                 "type": "claim", "report_position": "Erster Satz"},
                {"factoid": "Stromaufnahme 14kW typisch",
                 "type": "numeric_spec", "report_position": "Zweiter Satz"},
            ]),
            # verify_factoids_against_extracts
            "For each of the following factoids": json.dumps([
                {"factoid_index": 1, "verified": "true", "confidence": 0.9,
                 "supporting_extract_id": "E0",
                 "supporting_quote": "DGX B300 ist ein KI-Beschleuniger",
                 "reasoning": "Direkter Match"},
                {"factoid_index": 2, "verified": "true", "confidence": 0.92,
                 "supporting_extract_id": "E1",
                 "supporting_quote": "Stromaufnahme 14kW typisch",
                 "reasoning": "Direkter Match"},
            ]),
            # evaluate_coverage
            "research question is answered by the": json.dumps({
                "coverage": "answered", "confidence": 0.88,
                "missing_aspects": [], "supporting_extract_ids": ["E0", "E1"],
                "reasoning": "Zwei Extrakte mit hoher Verlässlichkeit decken die Frage.",
            }),
            # decide_continue_research
            "You control an autonomous research run": json.dumps({
                "decision": "stop_done", "confidence": 0.85,
                "next_round_focus": "",
                "reasoning": "Frage ist beantwortet, weitere Runde brächte nichts.",
            }),
            # diagnose_pipeline_state
            "Diagnose the state of this research run": json.dumps({
                "diagnosis": "successful", "confidence": 0.88,
                "remediation": "", "user_message": "Recherche erfolgreich",
                "reasoning": "Solide Quellenlage, niedrige Filter-Verlustrate.",
            }),
            # the harvest itself — last match entry (least specific)
            "extract relevant information from the following source": (
                "[F1] [POSITIV] Fakt: DGX B300 ist ein KI-Beschleuniger\n"
                "     Kontext: Produktbeschreibung\n"
                "     Verlässlichkeit: hoch\n"
                "[F1] [POSITIV] Fakt: Stromaufnahme 14kW typisch\n"
                "     Kontext: Power-Sektion\n"
                "     Verlässlichkeit: hoch\n"
            ),
        }

        llm = _ScriptedLLM(primary_responses=primary, harvest_responses=harvest)

        # ── 2. Connectors ──
        registry = ConnectorRegistry()

        search_results = [
            SearchResult(
                title="NVIDIA DGX B300 Datasheet",
                url="https://nvidia.com/dgx-b300/datasheet.pdf",
                snippet="DGX B300 is a flagship AI accelerator with 14kW TDP.",
                source_type=SourceType.WEB_SEARCH,
            ),
            SearchResult(
                title="ServeTheHome: DGX B300 Review",
                url="https://servethehome.com/dgx-b300-review",
                snippet="Detailed power testing.",
                source_type=SourceType.WEB_SEARCH,
            ),
        ]
        search_conn = _MockSearchConnector(search_results)
        registry.register(search_conn, is_search_engine=True)

        url_to_doc = {
            "https://nvidia.com/dgx-b300/datasheet.pdf": SourceDocument(
                SourceType.WEB_PAGE,
                "https://nvidia.com/dgx-b300/datasheet.pdf",
                "NVIDIA DGX B300 Datasheet",
                "DGX B300 specifications: 14kW TDP, 8x B300 GPUs. " * 30,
                {},
            ),
            "https://servethehome.com/dgx-b300-review": SourceDocument(
                SourceType.WEB_PAGE,
                "https://servethehome.com/dgx-b300-review",
                "ServeTheHome: DGX B300 Review",
                "Our review confirms 14kW typical power consumption. " * 30,
                {},
            ),
        }
        scraper = _MockWebScraper(url_to_doc)
        registry.register(scraper, is_web_scraper=True)

        # ── 3. Orchestrator ──
        config = PipelineConfig()
        config.max_rounds = 1  # a single round is enough for the smoke test
        config.data_dir = "/tmp/rh_smoke_test"

        orch = ResearchOrchestrator(
            llm=llm, connectors=registry, config=config,
        )

        progress_events = []
        async def progress(event_type, data=None):
            progress_events.append((event_type, data))

        # ── 4. Lauf ──
        ctx = await orch.run(
            query="Was ist die DGX B300 und wie viel Strom verbraucht sie?",
            chat_history=[],
            context_docs="",
            template_name="",
            progress_callback=progress,
            mode="web",
        )

        # ── 5. Acceptance ──
        # Core contracts: the pipeline ran through, the result is plausible.
        self.assertEqual(
            ctx.status, "done",
            f"wrong status: {ctx.status} — "
            f"error_message: {getattr(ctx, 'error_message', '?')}",
        )
        self.assertIsNotNone(ctx.research_plan)
        self.assertGreaterEqual(len(ctx.research_plan.questions), 1)
        self.assertGreater(len(ctx.sources), 0, "no sources fetched!")
        self.assertGreater(len(ctx.extracts), 0, "no extracts obtained!")

        # the classifier path ran
        self.assertGreater(
            len(ctx.coverage_per_round), 0,
            "the coverage classifier was not called — "
            "the classifier path is not active.",
        )
        # The continue decision is only logged in multi-round set-ups
        # (max_rounds=1: no need for a continue decision).
        # the diagnosis DID run
        self.assertIsNotNone(
            ctx.final_diagnosis,
            "the diagnosis classifier was not called — "
            "the diagnosis is not active.",
        )

        # report exists
        self.assertGreater(
            len(ctx.final_report), 50,
            "report suspiciously short",
        )

        # the factoid verification worked
        self.assertGreaterEqual(
            len(ctx.factoid_verifications), 1,
            "the factoid verification did not run.",
        )

    @async_test
    async def test_pipeline_runs_with_minimal_llm_responses(self):
        """Robustness: the pipeline also runs with mostly empty LLM answers.

        We do not check correctness here, but that nothing crashes when
        the LLM delivers nothing. Classifier fallbacks should apply, and
        the pipeline should end with ctx.status='done' (or at least a clean
        'error' status).
        """
        from src.config import PipelineConfig
        from src.pipeline.orchestrator import ResearchOrchestrator

        # Minimal set-up: the analysis phase delivers an empty plan, everything
        # else gets the LLM default = ""
        primary = {
            "You are a research planner": json.dumps({
                "summary": "Test",
                "questions": [{
                    "id": "F1", "question": "Test-Frage",
                    "search_terms": ["test"], "priority": "hoch",
                }],
                "direct_urls": [], "git_repos": [], "directory_queries": [],
            }),
            "Write the complete report": "Mini-Bericht",
        }
        # NO harvest answers — all classifiers will use their fallback

        llm = _ScriptedLLM(primary_responses=primary, harvest_responses={})
        registry = ConnectorRegistry()

        # a single source, to run the minimal pipeline
        search_results = [SearchResult(
            title="Test", url="https://test.com/page",
            snippet="Test snippet that is sufficiently long for the pipeline filters to accept it.",
            source_type=SourceType.WEB_SEARCH,
        )]
        registry.register(_MockSearchConnector(search_results), is_search_engine=True)
        registry.register(
            _MockWebScraper({}),  # default mock for all URLs
            is_web_scraper=True,
        )

        config = PipelineConfig()
        config.max_rounds = 1
        config.data_dir = "/tmp/rh_smoke_test_minimal"

        orch = ResearchOrchestrator(llm=llm, connectors=registry, config=config)

        async def progress(*args, **kwargs):
            pass

        ctx = await orch.run(
            query="Test", chat_history=[],
            context_docs="", template_name="",
            progress_callback=progress, mode="web",
        )

        # the pipeline may end with done or error — but must NOT crash
        self.assertIn(ctx.status, ("done", "error", "cancelled"))


if __name__ == "__main__":
    unittest.main()
