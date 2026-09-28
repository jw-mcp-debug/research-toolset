"""
End-to-end test of the bibliography check report: parse → database lookup
→ field comparison → report, with a fake LLM and fake databases. Runs the
report in English and in German (output-language catalog).
"""

import asyncio
import json

import pytest

from src import output_language
from src.connectors.literature_apis import LiteratureAPIClient
from src.pipeline.literature_check import LiteratureChecker

RAW = """Smith, J. (2020). Deep learning for bibliographies. Journal of Tests, 12(3), 1-10.

Doe, A. (2019). A paper nobody can find. Unknown Press."""

PARSED = [
    {"raw_text": "Smith, J. (2020). Deep learning for bibliographies.", "authors": ["Smith, J."],
     "title": "Deep learning for bibliographies", "year": "2020", "journal": "Journal of Tests",
     "volume": "12", "issue": "3", "pages": "1-10", "entry_type": "article"},
    {"raw_text": "Doe, A. (2019). A paper nobody can find.", "authors": ["Doe, A."],
     "title": "A paper nobody can find", "year": "2019", "entry_type": "book"},
]


class FakeLLM:
    async def primary_complete(self, messages, **kw):
        text = messages[-1]["content"]
        if "Context:" in text:
            return "Harmless formatting difference."
        return json.dumps(PARSED)

    def get_usage_stats(self):
        return {"total_requests": 2, "total_tokens": 100}


class FakeAPIs(LiteratureAPIClient):
    def __init__(self):
        super().__init__()

    async def lookup_entry(self, entry):
        if entry.title.startswith("Deep"):
            return [{"source": "crossref", "title": "Deep learning for bibliographies",
                     "authors": ["John Smith"], "year": "2021", "journal": "Journal of Tests",
                     "doi": "10.1000/test", "cited_by_count": 42}]
        return []


def _run(lang):
    checker = LiteratureChecker(FakeLLM(), searxng=None, lang=lang)
    checker.api_client = FakeAPIs()
    async def progress(*args, **kwargs):
        pass
    markdown, report = asyncio.run(checker.check(raw_text=RAW, progress_callback=progress))
    asyncio.run(checker.close())
    return markdown, report


def test_report_in_english():
    md, report = _run("en")
    assert report.total == 2
    assert "# Bibliography check report" in md
    assert "Entries checked" in md
    assert "Year" in md                     # deviation 2020 → 2021, field label
    assert "Corrected bibliography (APA)" in md
    assert "Prüfbericht" not in md


@pytest.fixture
def german(monkeypatch):
    monkeypatch.setenv("OUTPUT_LANGUAGES", "en,de")
    output_language.reload()
    yield
    monkeypatch.delenv("OUTPUT_LANGUAGES")
    output_language.reload()


def test_report_in_german(german):
    md, _ = _run("de")
    assert "# Prüfbericht Literaturverzeichnis" in md
    assert "Geprüfte Einträge" in md
    assert "Jahr" in md
    assert "Korrigiertes Literaturverzeichnis (APA)" in md


def test_llm_failure_is_not_reported_as_a_format_problem():
    """If the model cannot be reached, the message says so — not 'check the format'."""
    class BrokenLLM(FakeLLM):
        async def primary_complete(self, messages, **kw):
            raise ConnectionError("Connection error.")

    checker = LiteratureChecker(BrokenLLM(), searxng=None, lang="en")
    checker.api_client = FakeAPIs()

    async def progress(*args, **kwargs):
        pass
    md, report = asyncio.run(checker.check(raw_text=RAW, progress_callback=progress))
    asyncio.run(checker.close())
    assert report.total == 0
    assert "language model returned an error" in md and "Connection error" in md
    assert "check the format" not in md
