"""
Tests for the research mode table (single source of truth).

Guarantees:
  - Every mode ID resolves exactly to its route; unknown IDs fall back
    to web research.
  - Display labels are never used for routing: the UI routes on the
    stable ID, so labels can be translated without changing behaviour.
  - The institution mode is offered only with an institution profile.
  - ANALYSIS_MODE_MAP and ANALYSIS_USE_CASE_ORDER are derived from the
    same table and match the use-case registry.

Deliberately Gradio-free: `src.ui.research_modes` has no UI dependency.
"""

import ast
import unittest
from pathlib import Path

from src.institution import InstitutionProfile
from src.ui.research_modes import (
    ANALYSIS_MODE_MAP,
    ANALYSIS_USE_CASE_ORDER,
    DEFAULT_RESEARCH_MODE,
    RESEARCH_MODE_ORDER,
    RESEARCH_MODES,
    mode_choices,
    resolve_research_route,
)


class TestResearchModeTable(unittest.TestCase):

    def test_ids_unique(self):
        self.assertEqual(len(RESEARCH_MODE_ORDER), len(set(RESEARCH_MODE_ORDER)))

    def test_default_is_web(self):
        self.assertEqual(resolve_research_route(DEFAULT_RESEARCH_MODE), ("web", None))

    def test_every_id_resolves_exactly(self):
        for mode_id, _label, route in RESEARCH_MODES:
            self.assertEqual(resolve_research_route(mode_id), route, mode_id)

    def test_labels_do_not_resolve(self):
        # A display label must never act as a routing key.
        for _mode_id, label, _route in RESEARCH_MODES:
            if label.startswith("🌐"):
                continue
            self.assertEqual(resolve_research_route(label), ("web", None), label)

    def test_unknown_and_empty_default_to_web(self):
        for bogus in ["", None, "unknown", "Literatur", "legacy"]:
            self.assertEqual(resolve_research_route(bogus), ("web", None), bogus)

    def test_literature_modes_route_to_distinct_kinds(self):
        self.assertEqual(resolve_research_route("literature_check"), ("literature_check", None))
        self.assertEqual(resolve_research_route("literature_finder"), ("analysis", "literature_finder"))
        self.assertEqual(resolve_research_route("literature_review"), ("analysis", "literature_review"))


class TestModeChoices(unittest.TestCase):

    def test_without_profile_no_institution_mode(self):
        ids = [v for _, v in mode_choices(InstitutionProfile())]
        self.assertNotIn("institution", ids)
        self.assertEqual(ids[0], DEFAULT_RESEARCH_MODE)

    def test_with_profile_label_uses_short_name(self):
        prof = InstitutionProfile(name="Example University", short_name="EU",
                                  domains=("example.edu",))
        choices = dict((v, l) for l, v in mode_choices(prof))
        self.assertIn("institution", choices)
        self.assertIn("EU", choices["institution"])
        self.assertNotIn("{institution}", choices["institution"])

    def test_choices_are_label_id_pairs(self):
        for label, value in mode_choices(InstitutionProfile()):
            self.assertIn(value, RESEARCH_MODE_ORDER)
            self.assertIsInstance(label, str)


class TestUiRoutesOnIds(unittest.TestCase):
    """The UI must not compare mode values against label text."""

    def test_no_label_literals_compared_in_gradio_app(self):
        src = Path("src/ui/gradio_app.py").read_text()
        labels = {label for _, label, _ in RESEARCH_MODES}
        fragments = {"Tiefenerklärung", "Literatur", "Web-Recherche"}
        tree = ast.parse(src)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for c in [node.left, *node.comparators]:
                    if isinstance(c, ast.Constant) and isinstance(c.value, str) and (
                            c.value in labels or c.value in fragments):
                        offenders.append((node.lineno, c.value))
        self.assertEqual(offenders, [], "mode routing must use resolve_research_route()")


class TestDerivedStructures(unittest.TestCase):

    def test_analysis_map_keys_are_ids(self):
        for mode_id in ANALYSIS_MODE_MAP:
            self.assertIn(mode_id, RESEARCH_MODE_ORDER)

    def test_analysis_order_matches_table_order(self):
        expected = [uc for _, _, (kind, uc) in RESEARCH_MODES if kind == "analysis"]
        self.assertEqual(ANALYSIS_USE_CASE_ORDER, expected)

    def test_analysis_use_cases_exist_in_registry(self):
        from src.pipeline.analysis_pipeline import USE_CASE_REGISTRY
        for uc in ANALYSIS_USE_CASE_ORDER:
            self.assertIn(uc, USE_CASE_REGISTRY, uc)


if __name__ == "__main__":
    unittest.main()


class TestTemplates(unittest.TestCase):

    def test_template_ids_are_not_labels(self):
        from src.pipeline.models import DEFAULT_TEMPLATE, OUTPUT_TEMPLATES, template_choices
        self.assertIn(DEFAULT_TEMPLATE, OUTPUT_TEMPLATES)
        for label, tid in template_choices():
            self.assertIn(tid, OUTPUT_TEMPLATES)
            self.assertNotEqual(label, tid)
            self.assertRegex(tid, r"^[a-z_]+$")


class TestTwoLevelSelector(unittest.TestCase):
    """First dropdown: research forms + "Analyse …"; second: analysis modes."""

    def test_every_mode_reachable_exactly_once(self):
        from src.ui.research_modes import (
            ANALYSIS_GROUP, analysis_mode_choices, compose_mode, mode_group_choices,
        )
        prof = InstitutionProfile(name="Beispiel-Hochschule", domains=("example.org",))
        reachable = [compose_mode(v, "") for _, v in mode_group_choices(prof)
                     if v != ANALYSIS_GROUP]
        reachable += [compose_mode(ANALYSIS_GROUP, v)
                      for _, v in analysis_mode_choices(prof)]
        self.assertEqual(sorted(reachable), sorted(RESEARCH_MODE_ORDER))

    def test_group_value_is_not_a_mode_id(self):
        from src.ui.research_modes import ANALYSIS_GROUP
        self.assertNotIn(ANALYSIS_GROUP, RESEARCH_MODE_ORDER)

    def test_compose_falls_back_safely(self):
        from src.ui.research_modes import (
            ANALYSIS_GROUP, DEFAULT_ANALYSIS_MODE, compose_mode,
        )
        self.assertEqual(compose_mode(ANALYSIS_GROUP, "web"), DEFAULT_ANALYSIS_MODE)
        self.assertEqual(compose_mode("peer_review", ""), DEFAULT_RESEARCH_MODE)
        self.assertEqual(compose_mode(None, None), DEFAULT_RESEARCH_MODE)

    def test_every_mode_has_a_description(self):
        from src.ui.research_modes import mode_description
        prof = InstitutionProfile(name="Beispiel-Hochschule", domains=("example.org",))
        for mode_id in RESEARCH_MODE_ORDER:
            text = mode_description(mode_id, prof)
            self.assertTrue(text, mode_id)
            self.assertNotIn("{", text, mode_id)
