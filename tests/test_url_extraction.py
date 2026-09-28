"""
Tests for extracting URLs from user requests.

The analysis LLM sometimes overlooks URLs in the request, so primary
sources given by the user would not end up in direct_urls and the research
would miss its most important sources. The extraction acts as a
deterministic safety net.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Dependency mocks — as in the other tests
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
    class _AppConfig: pass
    class _PipelineConfig:
        max_rounds = 3
        max_parallel_fetches = 5
        max_content_chars = 50000
        min_content_chars = 100
        fetch_timeout_seconds = 30
        dedupe_threshold = 0.85
    class _SearXNGConfig:
        base_url = "http://test:8080"
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
    class _BaseConnector:
        name = "base"
    _base.BaseConnector = _BaseConnector
    sys.modules["src.connectors.base"] = _base
if "src.llm.client" not in sys.modules:
    _llm = MagicMock()
    _llm.DualLLMClient = MagicMock
    sys.modules["src.llm.client"] = _llm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import ResearchOrchestrator  # noqa: E402

extract = ResearchOrchestrator._extract_urls_from_text


def test_empty_input():
    """Empty or None input → empty list."""
    print("Test: empty input ...", end=" ")
    assert extract("") == []
    assert extract(None) == []
    assert extract("   ") == []
    print("✓")


def test_no_urls():
    """Text without URLs → empty list."""
    print("Test: text without URLs ...", end=" ")
    assert extract("Das ist ein ganz normaler Text ohne Links.") == []
    print("✓")


def test_single_url():
    """A single URL is recognised."""
    print("Test: single URL ...", end=" ")
    result = extract("Siehe https://example.com für Details.")
    assert result == ["https://example.com"]
    print("✓")


def test_original_bug_query():
    """A request with 4 URLs (comparison of several APIs)."""
    print("Test: Original-Bug-Query (4 URLs) ...", end=" ")
    query = (
        "Analysiere die unterschiedlichen Arten zur Api-Nutzung von "
        "Thinking bei Qwen 3.5 und Kimi 2.5. Hier sind die Links dazu "
        "https://huggingface.co/moonshotai/Kimi-K2.5\n"
        "https://huggingface.co/Qwen/Qwen3.5-397B-A17B\n"
        "https://huggingface.co/zai-org/GLM-5.1\n"
        "https://huggingface.co/google/gemma-4-31B-it"
    )
    result = extract(query)
    assert len(result) == 4, f"expected 4, got {len(result)}: {result}"
    assert "https://huggingface.co/moonshotai/Kimi-K2.5" in result
    assert "https://huggingface.co/google/gemma-4-31B-it" in result
    print("✓")


def test_trailing_punctuation_stripped():
    """Punctuation at the end of a URL is removed."""
    print("Test: Trailing-Punctuation ...", end=" ")
    assert extract("Siehe https://example.com.") == ["https://example.com"]
    assert extract("Mehr unter https://example.com,") == ["https://example.com"]
    assert extract("URL: https://example.com;") == ["https://example.com"]
    assert extract("URL: https://example.com\\") == ["https://example.com"]
    print("✓")


def test_url_in_parens():
    """A URL in brackets is returned without the bracket."""
    print("Test: URL in brackets ...", end=" ")
    assert extract("(siehe https://example.com)") == ["https://example.com"]
    print("✓")


def test_url_in_markdown_link():
    """A URL from a Markdown link is extracted."""
    print("Test: Markdown-Link ...", end=" ")
    assert extract("Schau [hier](https://example.com)") == ["https://example.com"]
    print("✓")


def test_query_string_preserved():
    """Query strings are kept completely."""
    print("Test: Query-String ...", end=" ")
    result = extract("API: https://api.example.com/v1?key=abc&id=123")
    assert result == ["https://api.example.com/v1?key=abc&id=123"]
    print("✓")


def test_anchor_preserved():
    """Anchors (#fragment) are kept."""
    print("Test: anchor fragment ...", end=" ")
    result = extract("Abschnitt https://docs.example.com/api#auth")
    assert result == ["https://docs.example.com/api#auth"]
    print("✓")


def test_http_and_https_both():
    """Both http and https are recognised."""
    print("Test: http and https ...", end=" ")
    result = extract("Alt: http://old.com und neu: https://new.com")
    assert "http://old.com" in result
    assert "https://new.com" in result
    print("✓")


def test_duplicates_removed():
    """Duplicates are removed (the first occurrence stays)."""
    print("Test: duplicates removed ...", end=" ")
    result = extract("https://a.com erst, dann https://a.com nochmal")
    assert result == ["https://a.com"]
    print("✓")


def test_order_preserved():
    """The order of appearance is kept."""
    print("Test: order ...", end=" ")
    result = extract("https://c.com dann https://a.com dann https://b.com")
    assert result == ["https://c.com", "https://a.com", "https://b.com"]
    print("✓")


def test_newlines_as_separators():
    """URLs over several lines are separated correctly."""
    print("Test: multi-line ...", end=" ")
    text = "Links:\nhttps://a.com\nhttps://b.com\nhttps://c.com"
    result = extract(text)
    assert len(result) == 3
    print("✓")


def main():
    print("=" * 60)
    print("URL extraction tests")
    print("=" * 60)
    tests = [
        test_empty_input, test_no_urls, test_single_url,
        test_original_bug_query, test_trailing_punctuation_stripped,
        test_url_in_parens, test_url_in_markdown_link,
        test_query_string_preserved, test_anchor_preserved,
        test_http_and_https_both, test_duplicates_removed,
        test_order_preserved, test_newlines_as_separators,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print("=" * 60)
    if failed:
        print(f"❌ {failed}/{len(tests)} tests failed")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed")


if __name__ == "__main__":
    main()


def test_user_url_detection_does_not_depend_on_the_reason_text():
    """A URL from the request counts as a user URL even if the analysis LLM
    put it into the plan itself (with its own reason text)."""
    import inspect
    src = inspect.getsource(ResearchOrchestrator._run_search_and_fetch)
    assert "User-Anfrage" not in src
    assert "_extract_urls_from_text(ctx.query" in src
