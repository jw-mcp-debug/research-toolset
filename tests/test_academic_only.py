"""
Tests for the academic-only mode.

Checks:
1. SearXNGConnector: the engines parameter is passed to the API correctly
2. SearXNGConnector: categories is NOT sent when engines is set
3. SearXNGConnector: the default stays categories="general"
4. ACADEMIC_ENGINES_WHITELIST contains the expected engines
5. orchestrator: the academic_only flag is set as self._academic_only
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Dependency mocks — as in test_orchestrator_plan_gate.py
if "httpx" not in sys.modules:
    _httpx = MagicMock()
    _httpx.TimeoutException = Exception
    sys.modules["httpx"] = _httpx
if "src.connectors.rate_limiter" not in sys.modules:
    _rl = MagicMock()
    _rl.get_rate_limiter = MagicMock(return_value=MagicMock())
    sys.modules["src.connectors.rate_limiter"] = _rl
if "src.config" not in sys.modules:
    _cfg = MagicMock()

    class _AppConfig:
        pass

    class _PipelineConfig:
        max_rounds = 3
        max_parallel_fetches = 5
        max_content_chars = 50000
        min_content_chars = 100
        fetch_timeout_seconds = 30
        dedupe_threshold = 0.85

    class _SearXNGConfig:
        base_url = "http://test-searxng:8080"
        timeout = 5
        max_results = 10
        language = "de-DE"

    _cfg.AppConfig = _AppConfig
    _cfg.PipelineConfig = _PipelineConfig
    _cfg.SearXNGConfig = _SearXNGConfig
    sys.modules["src.config"] = _cfg
if "src.connectors.base" not in sys.modules:
    _base = MagicMock()
    _base.ConnectorRegistry = MagicMock
    _base.normalize_url = lambda u: u
    _base.is_url_blocked = lambda u: False
    # BaseConnector must exist as a real class
    class _BaseConnector:
        name = "base"
    _base.BaseConnector = _BaseConnector
    sys.modules["src.connectors.base"] = _base
if "src.llm.client" not in sys.modules:
    _llm_mod = MagicMock()
    _llm_mod.DualLLMClient = MagicMock
    sys.modules["src.llm.client"] = _llm_mod

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


# ─── Tests ────────────────────────────────────────────────────────


async def test_engines_parameter_sent_to_api():
    """If engines is set, it goes to the SearXNG API as a parameter."""
    print("Test: engines-Parameter an API ...", end=" ")
    from src.connectors.searxng import SearXNGConnector

    # Mock-Config
    class _Config:
        base_url = "http://test:8080"
        timeout = 5
        max_results = 10
        language = "de-DE"

    conn = SearXNGConnector(_Config())

    # Mock-Client
    captured_params = {}

    async def _mock_get(url, params=None, **kwargs):
        captured_params.update(params or {})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {"results": []}
        return mock_resp

    conn.client = MagicMock()
    conn.client.get = _mock_get

    await conn.search(
        "test query",
        engines="wikipedia,google scholar,arxiv",
    )

    assert captured_params.get("engines") == "wikipedia,google scholar,arxiv", \
        f"engines missing or wrong: {captured_params}"
    # categories must NOT be present when engines is set
    assert "categories" not in captured_params, \
        f"categories must not be sent when engines is set: {captured_params}"
    print("✓")


async def test_categories_default_backward_compat():
    """Without the engines parameter, categories='general' is set as the default."""
    print("Test: categories default stays 'general' ...", end=" ")
    from src.connectors.searxng import SearXNGConnector

    class _Config:
        base_url = "http://test:8080"
        timeout = 5
        max_results = 10
        language = "de-DE"

    conn = SearXNGConnector(_Config())

    captured_params = {}

    async def _mock_get(url, params=None, **kwargs):
        captured_params.update(params or {})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {"results": []}
        return mock_resp

    conn.client = MagicMock()
    conn.client.get = _mock_get

    # no engines parameter set
    await conn.search("test query")

    assert captured_params.get("categories") == "general", \
        f"categories missing or wrong: {captured_params}"
    assert "engines" not in captured_params, \
        f"engines must not be sent without an explicit parameter: {captured_params}"
    print("✓")


async def test_explicit_categories_still_works():
    """An explicit categories='science' still works."""
    print("Test: explicit categories='science' ...", end=" ")
    from src.connectors.searxng import SearXNGConnector

    class _Config:
        base_url = "http://test:8080"
        timeout = 5
        max_results = 10
        language = "de-DE"

    conn = SearXNGConnector(_Config())

    captured_params = {}

    async def _mock_get(url, params=None, **kwargs):
        captured_params.update(params or {})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {"results": []}
        return mock_resp

    conn.client = MagicMock()
    conn.client.get = _mock_get

    await conn.search("test query", categories="science")

    assert captured_params.get("categories") == "science", \
        f"categories should be 'science': {captured_params}"
    assert "engines" not in captured_params
    print("✓")


async def test_engines_overrides_categories():
    """If engines AND categories are set, engines wins."""
    print("Test: engines overrides categories ...", end=" ")
    from src.connectors.searxng import SearXNGConnector

    class _Config:
        base_url = "http://test:8080"
        timeout = 5
        max_results = 10
        language = "de-DE"

    conn = SearXNGConnector(_Config())

    captured_params = {}

    async def _mock_get(url, params=None, **kwargs):
        captured_params.update(params or {})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: {"results": []}
        return mock_resp

    conn.client = MagicMock()
    conn.client.get = _mock_get

    # both set — engines should win
    await conn.search(
        "test query",
        categories="general",
        engines="wikipedia,arxiv",
    )

    assert captured_params.get("engines") == "wikipedia,arxiv"
    # categories must then NOT be sent
    assert "categories" not in captured_params, (
        f"categories must not be sent along when engines is set: "
        f"{captured_params}"
    )
    print("✓")


def test_academic_whitelist_contains_expected_engines():
    """ACADEMIC_ENGINES_WHITELIST contains the six expected engines."""
    print("Test: allow-list content ...", end=" ")
    from src.connectors.searxng import (
        ACADEMIC_ENGINES_WHITELIST,
        ACADEMIC_ENGINES_STRING,
    )

    expected = {
        "wikipedia",
        "wikidata",
        "google scholar",
        "semantic scholar",
        "arxiv",
        "pubmed",
    }
    assert set(ACADEMIC_ENGINES_WHITELIST) == expected, \
        f"allow list incomplete or has unexpected engines: {ACADEMIC_ENGINES_WHITELIST}"

    # the string variant must contain all engines
    for engine in expected:
        assert engine in ACADEMIC_ENGINES_STRING

    # no engines with spaces around the commas
    parts = ACADEMIC_ENGINES_STRING.split(",")
    assert all(p == p.strip() for p in parts), \
        f"allow-list string has untrimmed parts: {parts}"
    print("✓")


async def test_orchestrator_academic_only_flag_set():
    """orchestrator.run(academic_only=True) sets self._academic_only."""
    print("Test: Orchestrator academic_only-Flag ...", end=" ")

    # import the shared SmartMockLLM
    from tests._mocks import SmartMockLLM
    from src.pipeline.orchestrator import ResearchOrchestrator

    class _Connectors:
        def list(self):
            return []
        def get_connector_by_name(self, name):
            return None

    class _Config:
        max_rounds = 3
        max_parallel_fetches = 5
        max_content_chars = 50000
        min_content_chars = 100
        fetch_timeout_seconds = 30
        dedupe_threshold = 0.85

    orch = ResearchOrchestrator(
        llm=SmartMockLLM(),
        connectors=_Connectors(),
        config=_Config(),
        person_directory=None,
    )

    # Initial False
    assert orch._academic_only is False

    # After run() with academic_only=True the flag should be set.
    # We call run() with plan_only=True so that the pipeline does not
    # run completely.
    async def _silent(event, data):
        pass

    try:
        await orch.run(
            query="test",
            chat_history=[],
            context_docs="",
            template_name="Test",
            progress_callback=_silent,
            mode="peer_review",  # the analysis mode runs until decompose
            preflight_data={
                "paper_text": ("Abstract\n\n"
                              + "Test content hier. " * 200),
                "discipline": "Test",
                "role": "Gutachten",
                "focus": "",
            },
            plan_only=True,
            academic_only=True,
        )
    except Exception:
        # an error in decompose is fine — we only check the flag
        pass

    assert orch._academic_only is True, (
        f"academic_only should be True: {orch._academic_only}"
    )
    print("✓")


async def main():
    print("=" * 60)
    print("Academic-only mode tests")
    print("=" * 60)

    async_tests = [
        test_engines_parameter_sent_to_api,
        test_categories_default_backward_compat,
        test_explicit_categories_still_works,
        test_engines_overrides_categories,
        test_orchestrator_academic_only_flag_set,
    ]
    sync_tests = [
        test_academic_whitelist_contains_expected_engines,
    ]

    failed = 0
    total = len(async_tests) + len(sync_tests)

    for t in sync_tests:
        try:
            t()
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    for t in async_tests:
        try:
            await t()
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    if failed:
        print(f"❌ {failed}/{total} tests failed")
        sys.exit(1)
    else:
        print(f"✅ All {total} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
