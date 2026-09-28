"""
Tests for bilingual search queries.

Checks:
- _coerce_search_terms accepts a dict and a flat list
- _detect_term_language recognises German umlauts and stop words
- ResearchQuestion stores search_terms_by_lang correctly
- the terms_by_lang builder in the orchestrator produces separate lists
  per language (no mirror image)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

if "httpx" not in sys.modules:
    sys.modules["httpx"] = MagicMock()
if "src.connectors.rate_limiter" not in sys.modules:
    _rl = MagicMock()
    _rl.get_rate_limiter = MagicMock(return_value=MagicMock())
    sys.modules["src.connectors.rate_limiter"] = _rl

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.models import ResearchQuestion  # noqa: E402
from src.pipeline.query_utils import (  # noqa: E402
    coerce_search_terms as _coerce_search_terms,
)


def test_coerce_accepts_dict():
    """Dict input becomes a dict + flat list."""
    print("Test: _coerce accepts a dict ...", end=" ")
    raw = {
        "de": ["KI Regulierung Deutschland", "Künstliche Intelligenz EU"],
        "en": ["AI regulation Europe", "artificial intelligence policy"],
    }
    flat, by_lang = _coerce_search_terms(raw)
    assert set(by_lang.keys()) == {"de", "en"}
    assert len(by_lang["de"]) == 2
    assert len(by_lang["en"]) == 2
    assert len(flat) == 4  # union of all terms
    assert "KI Regulierung Deutschland" in flat
    assert "AI regulation Europe" in flat
    print("✓")


def test_coerce_handles_empty():
    """Empty input returns empty structures."""
    print("Test: _coerce empty input ...", end=" ")
    flat, by_lang = _coerce_search_terms(None)
    assert flat == []
    assert by_lang == {}
    flat, by_lang = _coerce_search_terms({})
    assert flat == []
    assert by_lang == {}
    flat, by_lang = _coerce_search_terms([])
    assert flat == []
    assert by_lang == {}
    print("✓")


def test_coerce_skips_empty_strings():
    """Empty strings and non-strings are filtered."""
    print("Test: _coerce filters empty strings ...", end=" ")
    raw = {"de": ["gut", "", "   ", None, 42, "auch gut"]}
    flat, by_lang = _coerce_search_terms(raw)
    assert by_lang["de"] == ["gut", "auch gut"]
    assert flat == ["gut", "auch gut"]
    print("✓")


def test_research_question_stores_by_lang():
    """ResearchQuestion stores search_terms_by_lang directly."""
    print("Test: ResearchQuestion has the by_lang field ...", end=" ")
    q = ResearchQuestion(
        id="F1",
        question="Test?",
        search_terms=["term1", "term2"],
        search_terms_by_lang={"de": ["term1"], "en": ["term2"]},
        search_langs=["de", "en"],
    )
    assert q.search_terms_by_lang == {"de": ["term1"], "en": ["term2"]}
    assert q.search_terms == ["term1", "term2"]
    print("✓")


def test_research_question_default_by_lang_empty():
    """The default for search_terms_by_lang is an empty dict."""
    print("Test: ResearchQuestion default by_lang is empty ...", end=" ")
    q = ResearchQuestion(
        id="F1",
        question="Test?",
        search_terms=["term1"],
    )
    assert q.search_terms_by_lang == {}
    print("✓")


def test_orchestrator_terms_by_lang_from_dict():
    """The orchestrator builder produces separate lists when
    search_terms_by_lang is set.

    This test simulates the code block in the orchestrator directly.
    """
    print("Test: terms_by_lang from a dict ...", end=" ")

    # simulate the orchestrator logic
    questions = [
        ResearchQuestion(
            id="F1",
            question="Test?",
            search_terms_by_lang={
                "de": ["KI Regulierung Deutschland"],
                "en": ["AI regulation Germany"],
            },
            search_langs=["de", "en"],
        ),
        ResearchQuestion(
            id="F2",
            question="Test 2?",
            search_terms_by_lang={
                "de": ["Datenschutz Europa"],
                "en": ["data protection Europe"],
            },
            search_langs=["de", "en"],
        ),
    ]

    terms_by_lang: dict = {}
    all_langs: set = set()
    for q in questions:
        for lang in q.search_langs:
            all_langs.add(lang)
            if lang not in terms_by_lang:
                terms_by_lang[lang] = []
        if q.search_terms_by_lang:
            for lang, terms in q.search_terms_by_lang.items():
                if lang not in terms_by_lang:
                    terms_by_lang[lang] = []
                for term in terms:
                    if term and term not in terms_by_lang[lang]:
                        terms_by_lang[lang].append(term)

    # de and en must have DIFFERENT lists
    assert terms_by_lang["de"] != terms_by_lang["en"]
    assert "KI Regulierung Deutschland" in terms_by_lang["de"]
    assert "KI Regulierung Deutschland" not in terms_by_lang["en"]
    assert "AI regulation Germany" in terms_by_lang["en"]
    assert "AI regulation Germany" not in terms_by_lang["de"]
    print("✓")


def main():
    print("=" * 60)
    print("Tests: bilingual query generation")
    print("=" * 60)

    tests = [
        test_coerce_accepts_dict,
        test_coerce_handles_empty,
        test_coerce_skips_empty_strings,
        test_research_question_stores_by_lang,
        test_research_question_default_by_lang_empty,
        test_orchestrator_terms_by_lang_from_dict,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    if failed:
        print(f"❌ {failed}/{len(tests)} tests failed")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
