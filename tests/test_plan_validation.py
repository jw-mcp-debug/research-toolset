"""
Tests for `src.pipeline.plan_validation`.

Acceptance:
  - conforming plan → empty issue list
  - plan without questions AND without alternatives → ERROR
  - plan without questions BUT with direct_urls → WARN
  - question without search_terms → ERROR
  - question without ID/text → ERROR
  - invalid priority → ERROR
  - invalid source_scope → WARN (not blocking)
  - unknown language code → WARN
  - duplicate question IDs → ERROR
  - has_blocking_errors filters correctly
"""

import unittest

from src.pipeline.models import (
    DirectURL,
    ResearchPlan,
    ResearchQuestion,
)
from src.pipeline.plan_validation import (
    PlanValidationIssue,
    format_issues_for_log,
    has_blocking_errors,
    validate_plan_conventions,
)


def _ok_plan() -> ResearchPlan:
    """Minimal conforming plan."""
    return ResearchPlan(
        summary="Knappe Zusammenfassung der Recherche-Ziele.",
        questions=[
            ResearchQuestion(
                id="F1",
                question="Was ist X?",
                search_terms=["X Definition"],
                priority="hoch",
                source_scope="web",
                search_langs=["de"],
            ),
        ],
    )


class TestConformPlan(unittest.TestCase):

    def test_minimal_conform_plan_has_no_issues(self):
        plan = _ok_plan()
        issues = validate_plan_conventions(plan)
        self.assertEqual(issues, [])
        self.assertFalse(has_blocking_errors(issues))

    def test_format_says_conforms_for_clean_plan(self):
        out = format_issues_for_log([])
        self.assertIn("conforms", out.lower())


class TestSummary(unittest.TestCase):

    def test_missing_summary_warns(self):
        plan = _ok_plan()
        plan.summary = ""
        issues = validate_plan_conventions(plan)
        # only a warning — research can run without a summary
        self.assertTrue(any(
            i.is_warning() and i.location == "plan.summary"
            for i in issues
        ))
        self.assertFalse(has_blocking_errors(issues))

    def test_too_short_summary_warns(self):
        plan = _ok_plan()
        plan.summary = "x"
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.category == "out_of_range" and "characters" in i.message
            for i in issues
        ))


class TestQuestionsCount(unittest.TestCase):

    def test_no_questions_no_alternatives_is_error(self):
        plan = ResearchPlan(summary="Test", questions=[])
        issues = validate_plan_conventions(plan)
        errors = [i for i in issues if i.is_error()]
        self.assertEqual(len(errors), 1)
        self.assertIn("the research has nothing to do", errors[0].message)

    def test_no_questions_but_direct_urls_only_warns(self):
        """Research can run with direct_urls alone — only a warning."""
        plan = ResearchPlan(
            summary="Test",
            questions=[],
            direct_urls=[DirectURL(url="https://example.com", reason="test")],
        )
        issues = validate_plan_conventions(plan)
        # The plan has 0 questions → exactly one issue about the number of questions
        question_count_issues = [
            i for i in issues if i.location == "plan.questions"
        ]
        self.assertEqual(len(question_count_issues), 1)
        self.assertTrue(question_count_issues[0].is_warning())
        self.assertFalse(has_blocking_errors(issues))

    def test_too_many_questions_warns(self):
        plan = _ok_plan()
        plan.questions = [
            ResearchQuestion(
                id=f"F{i}",
                question=f"Frage {i}?",
                search_terms=[f"term{i}"],
                priority="mittel",
            )
            for i in range(20)
        ]
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_warning() and "over-engineering" in i.message
            for i in issues
        ))


class TestQuestionFields(unittest.TestCase):

    def test_question_without_id_is_error(self):
        plan = _ok_plan()
        plan.questions[0].id = ""
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_error() and "without an ID" in i.message
            for i in issues
        ))

    def test_question_without_text_is_error(self):
        plan = _ok_plan()
        plan.questions[0].question = "  "
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_error() and i.location.endswith(".question")
            for i in issues
        ))

    def test_question_without_search_terms_is_error(self):
        plan = _ok_plan()
        plan.questions[0].search_terms = []
        plan.questions[0].search_terms_by_lang = {}
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_error() and "the search cannot start" in i.message
            for i in issues
        ))

    def test_search_terms_by_lang_alone_is_ok(self):
        """If only search_terms_by_lang is set and search_terms is empty: OK."""
        plan = _ok_plan()
        plan.questions[0].search_terms = []
        plan.questions[0].search_terms_by_lang = {"de": ["x"], "en": ["y"]}
        issues = validate_plan_conventions(plan)
        self.assertFalse(has_blocking_errors(issues))

    def test_invalid_priority_is_error(self):
        plan = _ok_plan()
        plan.questions[0].priority = "very_high"
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_error() and "priority" in i.message
            for i in issues
        ))

    def test_unknown_source_scope_is_warn(self):
        plan = _ok_plan()
        plan.questions[0].source_scope = "telegram"
        issues = validate_plan_conventions(plan)
        scope_issues = [
            i for i in issues if i.location.endswith(".source_scope")
        ]
        self.assertEqual(len(scope_issues), 1)
        self.assertTrue(scope_issues[0].is_warning())

    def test_too_many_search_terms_warns(self):
        plan = _ok_plan()
        plan.questions[0].search_terms = [f"t{i}" for i in range(15)]
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_warning() and i.location.endswith(".search_terms")
            for i in issues
        ))

    def test_unknown_lang_is_warn(self):
        plan = _ok_plan()
        plan.questions[0].search_langs = ["de", "klingon"]
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_warning() and "klingon" in i.message
            for i in issues
        ))

    def test_non_string_lang_is_error(self):
        plan = _ok_plan()
        plan.questions[0].search_langs = ["de", 42]
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_error() and "not a string" in i.message
            for i in issues
        ))


class TestDirectUrls(unittest.TestCase):

    def test_url_without_scheme_is_error(self):
        plan = _ok_plan()
        plan.direct_urls = [DirectURL(url="example.com", reason="x")]
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_error() and "http" in i.message
            for i in issues
        ))

    def test_empty_url_is_error(self):
        plan = _ok_plan()
        plan.direct_urls = [DirectURL(url="", reason="x")]
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_error() and "without a URL field" in i.message
            for i in issues
        ))

    def test_valid_url_no_issue(self):
        plan = _ok_plan()
        plan.direct_urls = [
            DirectURL(url="https://example.com/path", reason="x"),
        ]
        issues = validate_plan_conventions(plan)
        self.assertFalse(has_blocking_errors(issues))


class TestQuestionIdUniqueness(unittest.TestCase):

    def test_duplicate_ids_is_error(self):
        plan = _ok_plan()
        plan.questions = [
            ResearchQuestion(
                id="F1", question="?", search_terms=["x"], priority="hoch",
            ),
            ResearchQuestion(
                id="F1", question="?", search_terms=["y"], priority="hoch",
            ),
        ]
        issues = validate_plan_conventions(plan)
        self.assertTrue(any(
            i.is_error() and "duplicate question IDs" in i.message and "F1" in i.message
            for i in issues
        ))

    def test_empty_ids_dont_count_as_duplicates(self):
        """Several empty IDs → one error per question, no additional
        duplicates error."""
        plan = _ok_plan()
        plan.questions = [
            ResearchQuestion(id="", question="?", search_terms=["x"]),
            ResearchQuestion(id="", question="?", search_terms=["y"]),
        ]
        issues = validate_plan_conventions(plan)
        # There are errors for the two missing IDs, but no
        # "duplicate" error
        self.assertFalse(any("duplicate question IDs" in i.message for i in issues))


class TestHasBlockingErrors(unittest.TestCase):

    def test_warns_only_no_blocking(self):
        issues = [
            PlanValidationIssue("warn", "x", "msg1"),
            PlanValidationIssue("warn", "y", "msg2"),
        ]
        self.assertFalse(has_blocking_errors(issues))

    def test_one_error_blocks(self):
        issues = [
            PlanValidationIssue("warn", "x", "msg1"),
            PlanValidationIssue("error", "y", "msg2"),
        ]
        self.assertTrue(has_blocking_errors(issues))


class TestFormatIssuesForLog(unittest.TestCase):

    def test_error_marker(self):
        out = format_issues_for_log([
            PlanValidationIssue("error", "x", "Etwas Schlimmes"),
        ])
        self.assertIn("❌", out)
        self.assertIn("Etwas Schlimmes", out)

    def test_warn_marker(self):
        out = format_issues_for_log([
            PlanValidationIssue("warn", "x", "Etwas Komisches"),
        ])
        self.assertIn("⚠️", out)

    def test_location_appended(self):
        out = format_issues_for_log([
            PlanValidationIssue("error", "x", "msg",
                                location="plan.questions[0]"),
        ])
        self.assertIn("[plan.questions[0]]", out)


if __name__ == "__main__":
    unittest.main()
