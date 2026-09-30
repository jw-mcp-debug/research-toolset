"""
Tests for the UI helper modules.

Acceptance:
  - format_analysis_event handles all known event types
  - validate_explainer_inputs checks the four mandatory fields
  - build_explainer_preflight_data builds a consistent dict
  - format_task_plan_markdown returns a readable preview for a TaskPlan
  - the should_show_preview heuristics work correctly
  - extract_plan_metadata_info returns complete metadata
  - PreflightChecker validates context-sensitively
  - USE_CASE_REGISTRY has all analysis use cases with mandatory fields
"""

import unittest

from src.pipeline.analysis_pipeline import (
    PreflightChecker,
    Requirement,
    USE_CASE_REGISTRY,
)
from src.pipeline.models import (
    ResearchPlan,
    ResearchQuestion,
    SubTask,
    TaskPlan,
)
from src.ui.analysis_runner import (
    build_explainer_preflight_data,
    format_analysis_event,
    validate_explainer_inputs,
)
from src.ui.components.plan_preview import (
    extract_plan_metadata_info,
    format_task_plan_markdown,
    format_research_plan_markdown,
    should_show_preview,
    should_show_research_preview,
)


# ───────────────────────────────────────────────────────────────────
# format_analysis_event
# ───────────────────────────────────────────────────────────────────


class TestFormatAnalysisEvent(unittest.TestCase):

    def test_empty_event_returns_empty(self):
        self.assertEqual(format_analysis_event("", {}), "")

    def test_node_start(self):
        out = format_analysis_event("node_start", {"node": "analysis_layer_1"})
        self.assertIn("Start", out)
        self.assertIn("analysis_layer_1", out)

    def test_node_done_with_metadata(self):
        out = format_analysis_event("node_done", {
            "node": "layer_1",
            "metadata": {"executed": 3, "done": 3, "failed": 0},
        })
        self.assertIn("✅", out)
        self.assertIn("3/3 OK", out)

    def test_node_done_with_failures(self):
        out = format_analysis_event("node_done", {
            "node": "layer_2",
            "metadata": {"executed": 5, "done": 4, "failed": 1},
        })
        self.assertIn("4/5 OK", out)
        self.assertIn("1 fehlgeschlagen", out)

    def test_node_skipped(self):
        out = format_analysis_event("node_skipped", {
            "node": "format_agent", "reason": "Schema schon da",
        })
        self.assertIn("⊘", out)
        self.assertIn("format_agent", out)
        self.assertIn("Schema schon da", out)

    def test_node_failed_with_long_error_truncated(self):
        long_error = "x" * 500
        out = format_analysis_event("node_failed", {
            "node": "harvest", "error": long_error,
        })
        self.assertIn("❌", out)
        self.assertIn("…", out, "Lange Fehler werden mit Ellipse abgekürzt")
        self.assertLess(len(out), 300)

    def test_pipeline_done(self):
        out = format_analysis_event("pipeline_done", {"nodes": 15})
        self.assertIn("🏁", out)
        self.assertIn("15", out)

    def test_pipeline_stopped(self):
        out = format_analysis_event("pipeline_stopped", {})
        self.assertIn("⏹", out)

    def test_plan_ready(self):
        out = format_analysis_event("plan_ready", {"n_tasks": 8})
        self.assertIn("📋", out)
        self.assertIn("8 Aufgaben", out)

    def test_unknown_event_silent(self):
        out = format_analysis_event("never_heard_of", {"x": 1})
        self.assertEqual(out, "")


# ───────────────────────────────────────────────────────────────────
# validate_explainer_inputs / build_explainer_preflight_data
# ───────────────────────────────────────────────────────────────────


class TestExplainerInputs(unittest.TestCase):
    """The registry check must accept exactly what the explainer panel offers."""

    def test_valid_inputs(self):
        ok, msg = validate_explainer_inputs(
            topic="How does quantum entanglement work?",
            audience="Physics undergraduates without quantum mechanics",
            length="short",
            purpose="teaching",
        )
        self.assertTrue(ok, msg)
        self.assertEqual(msg, "")

    def test_missing_required_topic(self):
        ok, msg = validate_explainer_inputs(
            topic="", audience="Physics undergraduates, first year",
            length="short", purpose="",
        )
        self.assertFalse(ok)
        self.assertIn("Thema", msg)

    def test_audience_too_short(self):
        ok, msg = validate_explainer_inputs(
            topic="How does quantum entanglement work?", audience="Aliens",
            length="short", purpose="",
        )
        self.assertFalse(ok)
        self.assertIn("Zielgruppe", msg)

    def test_purpose_optional(self):
        """purpose is optional, an empty value is fine."""
        ok, msg = validate_explainer_inputs(
            topic="How does quantum entanglement work?",
            audience="Physics undergraduates, first year",
            length="short", purpose="",
        )
        self.assertTrue(ok, msg)

    def test_every_offered_choice_validates(self):
        """Every length/purpose the UI dropdown offers passes the check."""
        from src.pipeline.analysis_pipeline import (
            EXPLAINER_LENGTHS, EXPLAINER_PURPOSES,
        )
        for _, length in EXPLAINER_LENGTHS:
            for _, purpose in EXPLAINER_PURPOSES:
                ok, msg = validate_explainer_inputs(
                    "How does quantum entanglement work?",
                    "Physics undergraduates, first year", length, purpose,
                )
                self.assertTrue(ok, msg)

    def test_build_preflight_data_strips_values(self):
        d = build_explainer_preflight_data(
            topic="  Quantum mechanics  ",
            audience="Students",
            length="short",
            purpose="  teaching  ",
        )
        self.assertEqual(d["topic"], "Quantum mechanics")
        self.assertEqual(d["purpose"], "teaching")

    def test_build_preflight_data_drops_empty_optional(self):
        d = build_explainer_preflight_data(
            topic="x", audience="Students",
            length="short", purpose="",
        )
        self.assertNotIn("purpose", d,
                         "an empty optional field should not be in the dict")
        self.assertIn("topic", d)


# ───────────────────────────────────────────────────────────────────
# format_task_plan_markdown / format_research_plan_markdown
# ───────────────────────────────────────────────────────────────────


class TestFormatPlanMarkdown(unittest.TestCase):

    def test_none_plan(self):
        out = format_task_plan_markdown(None)
        self.assertIn("kein", out.lower())

    def test_empty_tasks(self):
        plan = TaskPlan(use_case="explainer", tasks=[])
        out = format_task_plan_markdown(plan)
        self.assertIn("explainer", out)

    def test_full_plan_lists_tasks_grouped_by_phase(self):
        plan = TaskPlan(use_case="explainer", tasks=[
            SubTask(id="K1", phase="concept",
                    description="Konzept extrahieren",
                    prompt_template="x"),
            SubTask(id="E1", phase="explanation",
                    description="Erklärung 1",
                    prompt_template="x", depends_on=["K1"]),
            SubTask(id="E2", phase="explanation",
                    description="Erklärung 2",
                    prompt_template="x", depends_on=["K1"]),
        ])
        out = format_task_plan_markdown(plan)
        self.assertIn("explainer", out)
        self.assertIn("3 Aufgaben", out)
        self.assertIn("2 Phasen", out)
        # The task IDs are in it
        self.assertIn("K1", out)
        self.assertIn("E1", out)
        # Dependency display
        self.assertIn("← K1", out)


class TestFormatResearchPlanMarkdown(unittest.TestCase):

    def test_delegates_to_plan_editor(self):
        """format_research_plan_markdown delegates to plan_editor."""
        plan = ResearchPlan(
            summary="Test-Plan-Zusammenfassung",
            questions=[ResearchQuestion(
                id="F1", question="Test?",
                search_terms=["test"], priority="hoch",
            )],
        )
        out = format_research_plan_markdown(plan)
        self.assertIn("Rechercheplan", out)
        self.assertIn("F1", out)
        self.assertIn("Test?", out)


# ───────────────────────────────────────────────────────────────────
# should_show_preview / should_show_research_preview
# ───────────────────────────────────────────────────────────────────


class TestShouldShowPreview(unittest.TestCase):

    def test_none_plan_never_shown(self):
        self.assertFalse(should_show_preview(None))

    def test_force_overrides(self):
        self.assertTrue(should_show_preview(None, force=True))

    def test_empty_plan_not_shown(self):
        plan = TaskPlan(use_case="x", tasks=[])
        self.assertFalse(should_show_preview(plan))

    def test_one_task_one_phase_not_shown(self):
        """Trivial plan — only 1 task, 1 phase: do not show."""
        plan = TaskPlan(use_case="x", tasks=[
            SubTask(id="T1", phase="p1", description="x",
                    prompt_template="x"),
        ])
        self.assertFalse(should_show_preview(plan))

    def test_three_or_more_tasks_shown(self):
        plan = TaskPlan(use_case="x", tasks=[
            SubTask(id=f"T{i}", phase="p1",
                    description="x", prompt_template="x")
            for i in range(3)
        ])
        self.assertTrue(should_show_preview(plan))

    def test_two_phases_shown(self):
        plan = TaskPlan(use_case="x", tasks=[
            SubTask(id="T1", phase="p1", description="x",
                    prompt_template="x"),
            SubTask(id="T2", phase="p2", description="x",
                    prompt_template="x"),
        ])
        self.assertTrue(should_show_preview(plan))


class TestShouldShowResearchPreview(unittest.TestCase):

    def test_none_not_shown(self):
        self.assertFalse(should_show_research_preview(None))

    def test_empty_plan_not_shown(self):
        self.assertFalse(should_show_research_preview(ResearchPlan()))

    def test_two_questions_shown(self):
        plan = ResearchPlan(questions=[
            ResearchQuestion(id="F1", question="?", search_terms=["x"]),
            ResearchQuestion(id="F2", question="?", search_terms=["y"]),
        ])
        self.assertTrue(should_show_research_preview(plan))

    def test_one_question_alone_not_shown(self):
        """Only 1 question and nothing else: trivial, do not show."""
        plan = ResearchPlan(questions=[
            ResearchQuestion(id="F1", question="?", search_terms=["x"]),
        ])
        self.assertFalse(should_show_research_preview(plan))

    def test_one_question_plus_url_shown(self):
        from src.pipeline.models import DirectURL
        plan = ResearchPlan(
            questions=[ResearchQuestion(
                id="F1", question="?", search_terms=["x"],
            )],
            direct_urls=[DirectURL(url="https://x.com", reason="test")],
        )
        self.assertTrue(should_show_research_preview(plan))


# ───────────────────────────────────────────────────────────────────
# extract_plan_metadata_info
# ───────────────────────────────────────────────────────────────────


class TestExtractPlanMetadataInfo(unittest.TestCase):

    def test_none_plan_empty_dict(self):
        self.assertEqual(extract_plan_metadata_info(None), {})

    def test_full_metadata(self):
        plan = TaskPlan(use_case="explainer", tasks=[
            SubTask(id="T1", phase="concept", description="x",
                    prompt_template="x"),
            SubTask(id="T2", phase="explanation", description="x",
                    prompt_template="x", depends_on=["T1"]),
            SubTask(id="T3", phase="explanation", description="x",
                    prompt_template="x", depends_on=["T1"]),
        ], estimated_calls=10, estimated_duration_seconds=120)

        info = extract_plan_metadata_info(plan)
        self.assertEqual(info["use_case"], "explainer")
        self.assertEqual(info["n_tasks"], 3)
        self.assertEqual(info["n_phases"], 2)
        self.assertEqual(info["phases"], ["concept", "explanation"])
        self.assertEqual(info["estimated_calls"], 10)
        self.assertEqual(info["estimated_duration_seconds"], 120)


# ───────────────────────────────────────────────────────────────────
# USE_CASE_REGISTRY and PreflightChecker
# ───────────────────────────────────────────────────────────────────


class TestUseCaseRegistry(unittest.TestCase):

    def test_all_seven_use_cases_registered(self):
        expected = {
            "explainer", "peer_review", "decision_analysis",
            "grant_proposal", "literature_review", "research_design",
            "literature_finder",
        }
        self.assertEqual(set(USE_CASE_REGISTRY.keys()), expected)

    def test_each_uc_has_preflight_with_requirements(self):
        for uc, cfg in USE_CASE_REGISTRY.items():
            self.assertIn("preflight", cfg, f"use case {uc} without preflight")
            checker = cfg["preflight"]
            reqs = checker.get_requirements()
            self.assertGreater(
                len(reqs), 0,
                f"use case {uc} has no mandatory fields",
            )

    def test_requirement_has_field_attribute(self):
        """gradio_app.py expects r.field for every requirement."""
        for cfg in USE_CASE_REGISTRY.values():
            for req in cfg["preflight"].get_requirements():
                self.assertTrue(hasattr(req, "field"))
                self.assertIsInstance(req.field, str)
                self.assertTrue(req.field, "field must not be empty")


class TestPreflightChecker(unittest.TestCase):

    def test_required_field_missing(self):
        checker = PreflightChecker([
            Requirement(field="x", label="X", required=True),
        ])
        ok, errors = checker.validate({})
        self.assertFalse(ok)
        self.assertEqual(len(errors), 1)

    def test_optional_field_missing_ok(self):
        checker = PreflightChecker([
            Requirement(field="x", label="X", required=False),
        ])
        ok, errors = checker.validate({})
        self.assertTrue(ok)

    def test_choice_invalid_value(self):
        checker = PreflightChecker([
            Requirement(field="x", label="X", kind="choice",
                        choices=["a", "b"]),
        ])
        ok, errors = checker.validate({"x": "c"})
        self.assertFalse(ok)
        self.assertTrue(any("erlaubten Optionen" in e for e in errors))

    def test_max_length_exceeded(self):
        checker = PreflightChecker([
            Requirement(field="x", label="X", max_length=5),
        ])
        ok, errors = checker.validate({"x": "viel zu lang"})
        self.assertFalse(ok)
        self.assertTrue(any("Zeichen" in e for e in errors))

    def test_whitespace_only_treated_as_empty(self):
        checker = PreflightChecker([
            Requirement(field="x", label="X", required=True),
        ])
        ok, errors = checker.validate({"x": "   "})
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
