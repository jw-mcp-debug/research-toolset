"""
Tests for the JSON conversion of the plan models (`from_dict`/`to_dict`
in `src.pipeline.models`).

JSON is the lingua franca between LLM output and the pipeline: the LLM
delivers plans as JSON, persistence writes them as JSON, and the pipeline
works with Python data classes. The conversion must live in one place
(the data model, not the UI), so that it is not reinvented inline in the
orchestrator.

Acceptance:
- round trip: `from_dict(to_dict(p))` yields an equivalent plan.
- defensive: broken/missing input does not crash, but yields defaults.
- practical format: a typical LLM output can be passed through without
  preprocessing.
"""

import json
import unittest

from src.pipeline.models import (
    DirectURL,
    GitRepoTarget,
    ResearchPlan,
    ResearchQuestion,
    DirectoryQuery,
)


# ── ResearchQuestion ──────────────────────────────────────────────


class TestResearchQuestionFromDict(unittest.TestCase):

    def test_full_dict(self):
        q = ResearchQuestion.from_dict({
            "id": "F1",
            "question": "Was kostet die DGX?",
            "search_terms": ["DGX Preis", "DGX price"],
            "search_langs": ["de", "en"],
            "source_scope": "web",
            "priority": "hoch",
        })
        self.assertEqual(q.id, "F1")
        self.assertEqual(q.question, "Was kostet die DGX?")
        self.assertEqual(q.search_terms, ["DGX Preis", "DGX price"])
        self.assertEqual(q.priority, "high")  # normalised code

    def test_missing_id_uses_fallback(self):
        q = ResearchQuestion.from_dict(
            {"question": "Test"}, fallback_id="F7",
        )
        self.assertEqual(q.id, "F7")

    def test_search_terms_dict_format_normalized(self):
        """The LLM may deliver search_terms as a dict per language."""
        q = ResearchQuestion.from_dict({
            "id": "F1", "question": "?",
            "search_terms": {"de": ["term1"], "en": ["term2"]},
        })
        # the flat list contains both
        self.assertIn("term1", q.search_terms)
        self.assertIn("term2", q.search_terms)
        # search_terms_by_lang is filled
        self.assertEqual(q.search_terms_by_lang.get("de"), ["term1"])

    def test_invalid_input_returns_empty_question(self):
        # string instead of dict
        q = ResearchQuestion.from_dict("kaputt")
        self.assertEqual(q.id, "")
        self.assertEqual(q.question, "")

    def test_round_trip(self):
        original = ResearchQuestion(
            id="F1", question="Was?",
            search_terms=["a", "b"],
            priority="mittel",
        )
        roundtrip = ResearchQuestion.from_dict(original.to_dict())
        self.assertEqual(roundtrip.id, "F1")
        self.assertEqual(roundtrip.question, "Was?")
        self.assertEqual(roundtrip.search_terms, ["a", "b"])
        self.assertEqual(roundtrip.priority, "medium")  # normalised code


# ── DirectURL / GitRepoTarget / DirectoryQuery ──────────────────────────


class TestDirectURL(unittest.TestCase):

    def test_round_trip(self):
        original = DirectURL(url="https://example.com", reason="Primärquelle")
        rt = DirectURL.from_dict(original.to_dict())
        self.assertEqual(rt.url, original.url)
        self.assertEqual(rt.reason, original.reason)

    def test_url_trailing_chars_stripped(self):
        """LLM output sometimes has ),. at the end of a URL."""
        u = DirectURL.from_dict({"url": "https://example.com),", "reason": ""})
        self.assertEqual(u.url, "https://example.com")

    def test_invalid_input(self):
        u = DirectURL.from_dict("kaputt")
        self.assertEqual(u.url, "")


class TestGitRepoTarget(unittest.TestCase):

    def test_round_trip(self):
        original = GitRepoTarget(
            owner="x", repo="y", platform="github",
            search_terms=["a", "b"],
        )
        rt = GitRepoTarget.from_dict(original.to_dict())
        self.assertEqual(rt.owner, "x")
        self.assertEqual(rt.search_terms, ["a", "b"])

    def test_default_platform(self):
        gr = GitRepoTarget.from_dict({"owner": "a", "repo": "b"})
        self.assertEqual(gr.platform, "github")


class TestDirectoryQuery(unittest.TestCase):

    def test_llm_output_uses_query_field(self):
        """The LLM delivers {"query": "...", "reason": "..."}."""
        zq = DirectoryQuery.from_dict({
            "query": "Hans Müller", "reason": "Person gesucht",
        })
        self.assertEqual(zq.query, "Hans Müller")
        self.assertEqual(zq.reason, "Person gesucht")

    def test_persisted_uses_query(self):
        """Persisted plans store the search text under `query`."""
        zq = DirectoryQuery.from_dict({"query": "X", "reason": "y"})
        self.assertEqual(zq.query, "X")

    def test_round_trip(self):
        original = DirectoryQuery(query="X", reason="y", subject_area="IT")
        rt = DirectoryQuery.from_dict(original.to_dict())
        self.assertEqual(rt.query, "X")
        self.assertEqual(rt.subject_area, "IT")


# ── ResearchPlan ─────────────────────────────────────────────────


class TestResearchPlanFromDict(unittest.TestCase):

    def test_realistic_llm_output(self):
        """The typical LLM output format is accepted without preprocessing."""
        llm_output = {
            "summary": "Recherche zu DGX",
            "questions": [
                {"id": "F1", "question": "Preis?",
                 "search_terms": ["DGX Preis", "DGX price"],
                 "search_langs": ["de", "en"],
                 "source_scope": "web",
                 "priority": "hoch"},
                {"id": "F2", "question": "Stromverbrauch?",
                 "search_terms": ["DGX Strom"],
                 "priority": "mittel"},
            ],
            "direct_urls": [{"url": "https://nvidia.com", "reason": "OEM"}],
            "git_repos": [],
            "directory_queries": [],
        }
        plan = ResearchPlan.from_dict(llm_output)
        self.assertEqual(len(plan.questions), 2)
        self.assertEqual(plan.questions[0].id, "F1")
        self.assertEqual(len(plan.direct_urls), 1)
        self.assertEqual(plan.summary, "Recherche zu DGX")

    def test_invalid_input_returns_empty(self):
        for bad in [None, "string", 42, [1, 2, 3]]:
            plan = ResearchPlan.from_dict(bad)
            self.assertEqual(plan.summary, "")
            self.assertEqual(len(plan.questions), 0)

    def test_invalid_question_entries_dropped(self):
        """Broken question entries are discarded, good ones kept."""
        plan = ResearchPlan.from_dict({
            "questions": [
                {"id": "F1", "question": "Gut"},
                "kaputt",       # string instead of dict → empty → discarded
                {},             # empty dict → empty → discarded
                {"id": "F4", "question": "OK"},
            ],
        })
        self.assertEqual(len(plan.questions), 2)
        self.assertEqual(plan.questions[0].id, "F1")
        self.assertEqual(plan.questions[1].id, "F4")

    def test_empty_url_dropped(self):
        plan = ResearchPlan.from_dict({
            "direct_urls": [
                {"url": "", "reason": "leer"},
                {"url": "https://valid.com", "reason": "ok"},
            ],
        })
        self.assertEqual(len(plan.direct_urls), 1)
        self.assertEqual(plan.direct_urls[0].url, "https://valid.com")

    def test_round_trip_preserves_structure(self):
        original = ResearchPlan(
            summary="Test",
            questions=[
                ResearchQuestion(id="F1", question="?",
                                 search_terms=["a"], priority="hoch"),
            ],
            direct_urls=[DirectURL(url="https://x.de", reason="ok")],
            git_repos=[GitRepoTarget(owner="o", repo="r")],
            directory_queries=[DirectoryQuery(query="Q")],
        )
        rt = ResearchPlan.from_dict(original.to_dict())
        self.assertEqual(rt.summary, original.summary)
        self.assertEqual(len(rt.questions), 1)
        self.assertEqual(rt.questions[0].id, "F1")
        self.assertEqual(len(rt.direct_urls), 1)
        self.assertEqual(len(rt.git_repos), 1)
        self.assertEqual(len(rt.directory_queries), 1)

    def test_json_string_round_trip(self):
        """The plan can be serialised as JSON and read back (the typical
        persistence path)."""
        original = ResearchPlan(
            summary="Test",
            questions=[ResearchQuestion(id="F1", question="?")],
        )
        js = json.dumps(original.to_dict(), ensure_ascii=False)
        rt = ResearchPlan.from_dict(json.loads(js))
        self.assertEqual(rt.summary, "Test")
        self.assertEqual(rt.questions[0].id, "F1")


if __name__ == "__main__":
    unittest.main()
