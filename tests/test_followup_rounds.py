"""
Integration test: a research run with several rounds actually searches again.

Round 1 searches the plan's terms. The coverage classifier reports missing
aspects and the continue decision asks for another round; round 2 must then
search for those aspects and fetch new sources.
"""

import json
import re

from tests.test_pipeline_smoke import _MockWebScraper, _ScriptedLLM  # noqa: F401  (test isolation preamble)
from src.connectors.base import BaseConnector, ConnectorRegistry
from src.pipeline.models import SearchResult, SourceType


class _QueryAwareSearch(BaseConnector):
    """Returns one result per query, with a URL derived from the query."""
    name = "query_aware_search"

    def __init__(self):
        self.queries: list[str] = []

    async def search(self, query, max_results=10, **kwargs):
        self.queries.append(query)
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
        return [SearchResult(title=f"Result for {query}", url=f"https://example.org/{slug}",
                             snippet=f"About {query}.", source_type=SourceType.WEB_SEARCH)]

    async def fetch(self, url):
        return None

    def can_handle(self, url):
        return False


PLAN = {
    "summary": "Power draw of the DGX B300",
    "questions": [{
        "id": "F1", "question": "How much power does the DGX B300 draw?",
        "search_terms": {"en": ["DGX B300 power"]}, "search_langs": ["en"],
        "source_scope": "web", "priority": "high",
    }],
}


async def _run(max_rounds):
    from src.config import PipelineConfig
    from src.pipeline.orchestrator import ResearchOrchestrator

    llm = _ScriptedLLM(
        primary_responses={"You are a research planner": json.dumps(PLAN)},
        harvest_responses={
            "PERSON or ORGANISATION ANCHOR": json.dumps(
                {"anchor_type": "none", "target": "", "confidence": 0.95, "reasoning": "product question"}),
            "research question is answered by the": json.dumps(
                {"coverage": "partial", "confidence": 0.8, "missing_aspects": ["idle power"],
                 "supporting_extract_ids": [], "reasoning": "peak power found, idle power missing"}),
            "You control an autonomous research run": json.dumps(
                {"decision": "continue", "confidence": 0.9, "next_round_focus": "idle power",
                 "reasoning": "idle power still missing"}),
            "Diagnose the state of this research run": json.dumps(
                {"diagnosis": "partial_success", "confidence": 0.8, "remediation": "",
                 "user_message": "ok", "reasoning": "-"}),
            "Identify all FACTOIDS": "[]",
            "extract relevant information from the following source": (
                "[F1] [POSITIVE] Fact: The DGX B300 draws about 14 kW\n"
                "     Context: datasheet\n     Reliability: high"),
        },
    )
    search = _QueryAwareSearch()
    registry = ConnectorRegistry()
    registry.register(search, is_search_engine=True)
    registry.register(_MockWebScraper({}), is_web_scraper=True)
    config = PipelineConfig()
    config.max_rounds = max_rounds
    config.data_dir = "/tmp/followup_rounds_test"
    orch = ResearchOrchestrator(llm=llm, connectors=registry, config=config)

    async def progress(*_a, **_k):
        pass

    ctx = await orch.run(query="How much power does the DGX B300 draw?", chat_history=[],
                         context_docs="", template_name="", progress_callback=progress, mode="web")
    return ctx, search


async def test_second_round_searches_for_the_missing_aspects():
    ctx, search = await _run(max_rounds=2)
    assert ctx.rounds_completed == 2
    assert any("idle power" in q for q in search.queries), search.queries
    urls = {s.url for s in ctx.sources}
    assert any("idle-power" in u for u in urls), urls


async def test_single_round_configuration_does_not_search_again():
    ctx, search = await _run(max_rounds=1)
    assert ctx.rounds_completed == 1
    assert not any("idle power" in q for q in search.queries)
