"""
Tests for `src.ui.plan_editor`.

Acceptance: the Markdown preview of a plan is human-readable and contains
all relevant plan fields.
"""

import unittest

from src.pipeline.models import (
    DirectURL,
    GitRepoTarget,
    ResearchPlan,
    ResearchQuestion,
    DirectoryQuery,
)
from src.ui.plan_editor import format_plan_markdown


def _full_plan() -> ResearchPlan:
    """Plan with all field types for round-trip tests."""
    return ResearchPlan(
        summary="Recherche zu DGX-Hardware",
        questions=[
            ResearchQuestion(
                id="F1",
                question="Was kostet die DGX B300?",
                search_terms=["DGX B300 price"],
                search_terms_by_lang={
                    "en": ["DGX B300 price"],
                    "de": ["DGX B300 Preis"],
                },
                priority="hoch",
                source_scope="web",
                search_langs=["de", "en"],
            ),
            ResearchQuestion(
                id="F2",
                question="Stromverbrauch?",
                search_terms=["DGX B300 TDP"],
                priority="mittel",
            ),
        ],
        direct_urls=[
            DirectURL(url="https://nvidia.com/dgx-b300",
                      reason="Hersteller-Datasheet"),
        ],
        git_repos=[
            GitRepoTarget(
                owner="NVIDIA", repo="cuda-samples",
                platform="github", search_scope="readme",
                search_terms=["B300"],
            ),
        ],
        directory_queries=[
            DirectoryQuery(
                query_type="search", query="Uni-Computing",
                reason="Uni-interne Hardware-Beschaffung",
                pid=42, with_children=True, with_subobjects=False,
                subject_area="IT", org_filter="example.edu",
            ),
        ],
        followup_queries=[{"query": "follow-up Beispiel"}],
    )


# ───────────────────────────────────────────────────────────────────
# Markdown-Format
# ───────────────────────────────────────────────────────────────────


class TestFormatPlanMarkdown(unittest.TestCase):

    def test_none_plan(self):
        out = format_plan_markdown(None)
        self.assertIn("kein Plan", out)

    def test_empty_plan(self):
        out = format_plan_markdown(ResearchPlan())
        self.assertIn("Rechercheplan", out)
        # No sections for empty fields
        self.assertNotIn("### Questions", out)
        self.assertNotIn("### Direct URLs", out)

    def test_full_plan_contains_all_sections(self):
        out = format_plan_markdown(_full_plan())
        self.assertIn("Rechercheplan", out)
        self.assertIn("Recherche zu DGX-Hardware", out)
        # Questions
        self.assertIn("Fragen (2)", out)
        self.assertIn("**[F1]**", out)
        self.assertIn("Was kostet die DGX B300?", out)
        self.assertIn("**[F2]**", out)
        # Direct URLs
        self.assertIn("Direkte URLs", out)
        self.assertIn("https://nvidia.com/dgx-b300", out)
        self.assertIn("Hersteller-Datasheet", out)
        # Git
        self.assertIn("Git-Repos", out)
        self.assertIn("NVIDIA/cuda-samples", out)
        # person directory
        self.assertIn("Abfragen im Personenverzeichnis (1)", out)
        self.assertIn("Uni-Computing", out)

    def test_question_with_search_terms_by_lang(self):
        plan = ResearchPlan(
            questions=[ResearchQuestion(
                id="F1", question="x",
                search_terms_by_lang={"en": ["a", "b"], "de": ["c"]},
                priority="hoch",
            )],
        )
        out = format_plan_markdown(plan)
        self.assertIn("`en`: a, b", out)
        self.assertIn("`de`: c", out)


