"""
Tests for `src.pipeline.filters.base` + `person_match`.

Acceptance:
  - the PersonHallucinationFilter activates ONLY if query_anchor.is_person()
  - with type='none' (subject-matter query): the filter does NOT become
    active → no extracts discarded
  - with type='person' and a match: extracts are kept
  - with type='person' and no match: extracts are discarded
  - FilterStats are computed correctly (loss_rate)
"""

import unittest

from src.pipeline.classifiers.query_anchor import QueryAnchor
from src.pipeline.filters.base import (
    FilterStats,
    run_filter_pipeline,
)
from src.pipeline.filters.person_match import (
    PersonHallucinationFilter,
    is_person_in_source,
    normalize_for_match,
)
from tests._helpers import StubContext, async_test


# ═══════════════════════════════════════════════════════════════════
# is_person_in_source — the deterministic token-match function
# ═══════════════════════════════════════════════════════════════════


class TestIsPersonInSource(unittest.TestCase):

    def test_full_name_substring(self):
        content = "Jonas Brenner hat einen Vortrag gehalten."
        self.assertTrue(is_person_in_source("Jonas Brenner", content))

    def test_separated_tokens(self):
        # tokens not next to each other, but all present
        content = "Brenner berichtet... [später] ... Jonas schließt ab"
        self.assertTrue(is_person_in_source("Jonas Brenner", content))

    def test_no_match(self):
        content = "Ein anderer Vortrag von Hans Müller."
        self.assertFalse(is_person_in_source("Jonas Brenner", content))

    def test_diacritics_normalized(self):
        # Müller in the source, search for Mueller — should match
        content = "Prof. Müller hat das ausgeführt."
        self.assertTrue(is_person_in_source("Prof. Mueller", content))

    def test_empty_target_returns_true(self):
        # safe default: empty target → no filtering
        self.assertTrue(is_person_in_source("", "irgendwas"))

    def test_case_insensitive(self):
        content = "JONAS BRENNER hielt einen Vortrag."
        self.assertTrue(is_person_in_source("Jonas Brenner", content))

    def test_dr_prefix_handled(self):
        # "Dr. Schmidt" vs "Schmidt" in the source
        content = "Prof. Schmidt erläuterte..."
        self.assertTrue(is_person_in_source("Dr. Schmidt", content))

    def test_dgx_components_dont_match(self):
        """Product query at the token-match level.

        This is the SECOND line of defence: even if someone activates the
        filter wrongly (e.g. with target='Preis-Leistungs-Verhältnis'), a
        datasheet source that does not contain this complete string must
        NOT be discarded completely if the tokens "Preis", "Leistungs",
        "Verhältnis" occur in it individually.

        Admittedly this is not the primary shield — primary is the
        classifier, which does not set type='person' in the first place.
        But if someone forces the filter anyway (e.g. via a use-case
        override), at least not everything containing the words should go.
        """
        nvidia_content = (
            "DGX B300 specifications: 14kW power consumption. "
            "Excellent performance ratio. Listing price: $500k."
        )
        # Match logic: all tokens must occur. For "Preis-Leistungs-
        # Verhältnis" the tokens are "preis", "leistungs", "verhaltnis"
        # (after normalisation). Of these, "preis" and "leistungs" are not
        # found. Expectation: no match.
        self.assertFalse(is_person_in_source(
            "Preis-Leistungs-Verhältnis", nvidia_content,
        ))


# ═══════════════════════════════════════════════════════════════════
# PersonHallucinationFilter — applies_to + apply
# ═══════════════════════════════════════════════════════════════════


class TestPersonHallucinationFilter(unittest.TestCase):

    def _make_filter(self, sources_by_url):
        return PersonHallucinationFilter(
            source_content_lookup=sources_by_url.get,
        )

    def test_applies_to_person_with_high_confidence(self):
        anchor = QueryAnchor(type="person", target="Jonas Brenner",
                             confidence=0.91)
        ctx = StubContext(query_anchor=anchor)
        f = self._make_filter({})

        active, reason = f.applies_to(ctx)
        self.assertTrue(active)
        self.assertIn("Jonas Brenner", reason)

    def test_does_not_apply_to_none_anchor(self):
        """Subject-matter query: no person anchor → filter inactive."""
        anchor = QueryAnchor(type="none", target="", confidence=0.94)
        ctx = StubContext(query_anchor=anchor)
        f = self._make_filter({})

        active, reason = f.applies_to(ctx)
        self.assertFalse(active)
        self.assertIn("not a person", reason)

    def test_does_not_apply_below_threshold(self):
        """Confidence 0.65 < 0.7 → filter inactive (safe default)."""
        anchor = QueryAnchor(type="person", target="Müller",
                             confidence=0.65)
        ctx = StubContext(query_anchor=anchor)
        f = self._make_filter({})

        active, _ = f.applies_to(ctx)
        self.assertFalse(active)

    def test_does_not_apply_without_anchor(self):
        ctx = StubContext(query_anchor=None)
        f = self._make_filter({})

        active, reason = f.applies_to(ctx)
        self.assertFalse(active)
        self.assertIn("no query_anchor", reason)

    @async_test
    async def test_apply_keeps_matching_sources(self):
        anchor = QueryAnchor(type="person", target="Jonas Brenner",
                             confidence=0.9)
        ctx = StubContext(query_anchor=anchor)

        sources = {
            "https://uni.example/brenner-bio":
                "Prof. Jonas Brenner hat 20 Publikationen.",
            "https://other.example/zoo":
                "Ein Zooführer und Tierpfleger.",
        }
        f = self._make_filter(sources)

        extracts = [
            {"source_url": "https://uni.example/brenner-bio",
             "fact": "20 Publikationen", "question_id": "F1"},
            {"source_url": "https://uni.example/brenner-bio",
             "fact": "Mitglied im Beirat", "question_id": "F1"},
            {"source_url": "https://other.example/zoo",
             "fact": "Halluzination", "question_id": "F1"},
        ]

        kept, rejected, stats = await f.apply(extracts, ctx)

        self.assertEqual(len(kept), 2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(stats.total, 3)
        self.assertEqual(stats.kept, 2)
        self.assertEqual(stats.rejected, 1)
        self.assertAlmostEqual(stats.loss_rate, 1/3, places=2)

    @async_test
    async def test_apply_with_no_target_keeps_all(self):
        """Safety net: empty target → nothing is discarded."""
        anchor = QueryAnchor(type="person", target="", confidence=0.99)
        ctx = StubContext(query_anchor=anchor)
        f = self._make_filter({"u": "irgendwas"})

        extracts = [{"source_url": "u", "fact": "x", "question_id": "F1"}]
        kept, rejected, stats = await f.apply(extracts, ctx)

        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 0)


# ═══════════════════════════════════════════════════════════════════
# run_filter_pipeline — the filter architecture as a whole
# ═══════════════════════════════════════════════════════════════════


class TestFilterPipeline(unittest.TestCase):

    @async_test
    async def test_dgx_scenario_no_filter_active(self):
        """END TO END for a subject-matter query.

        Set-up: anchor='none' (the classifier recognised it correctly).
        Expectation: PersonHallucinationFilter not active → all 528
        extracts are kept → no empty report.
        """
        anchor = QueryAnchor(
            type="none", target="", confidence=0.94,
            reasoning="Vergleichsanfrage zu Hardware-Spezifikationen",
        )
        ctx = StubContext(query_anchor=anchor)

        # Simulate 528 extracts from 26 sources — realistic
        # figures
        sources = {f"https://nvidia.com/source-{i}":
                   "DGX B300 datasheet content..." for i in range(26)}
        extracts = []
        for src_idx in range(26):
            for ex_idx in range(528 // 26 + 1):
                extracts.append({
                    "source_url": f"https://nvidia.com/source-{src_idx}",
                    "fact": f"Spec {src_idx}-{ex_idx}",
                    "question_id": "F1",
                })
        extracts = extracts[:528]

        f = PersonHallucinationFilter(source_content_lookup=sources.get)

        kept, stats_map = await run_filter_pipeline(
            items=extracts, ctx=ctx, filters=[f],
        )

        # CORE: all 528 extracts are kept
        self.assertEqual(len(kept), 528)
        # the filter stats show: not active
        self.assertFalse(stats_map["person_hallucination"].activated)
        self.assertEqual(stats_map["person_hallucination"].rejected, 0)

    @async_test
    async def test_legitimate_person_research_filter_works(self):
        """A genuine person search: the filter separates correctly."""
        anchor = QueryAnchor(type="person", target="Jonas Brenner",
                             confidence=0.91)
        ctx = StubContext(query_anchor=anchor)

        # 4 sources: 2 mention the person, 2 do not
        sources = {
            "https://a/1": "Jonas Brenner hat das untersucht.",
            "https://a/2": "Brenner und sein Team. Jonas ist Erstautor.",
            "https://b/1": "Hans Schmidt war hier.",
            "https://b/2": "Marie Curie und Pierre Curie.",
        }
        extracts = [
            {"source_url": url, "fact": f"x-{url}", "question_id": "F1"}
            for url in sources
        ]

        f = PersonHallucinationFilter(source_content_lookup=sources.get)

        kept, stats_map = await run_filter_pipeline(
            items=extracts, ctx=ctx, filters=[f],
        )

        self.assertEqual(len(kept), 2)
        kept_urls = {e["source_url"] for e in kept}
        self.assertEqual(kept_urls, {"https://a/1", "https://a/2"})

        s = stats_map["person_hallucination"]
        self.assertTrue(s.activated)
        self.assertEqual(s.total, 4)
        self.assertEqual(s.kept, 2)
        self.assertEqual(s.rejected, 2)
        self.assertEqual(s.loss_rate, 0.5)


class TestFilterStats(unittest.TestCase):

    def test_loss_rate_zero_division_safe(self):
        s = FilterStats(name="x", total=0)
        self.assertEqual(s.loss_rate, 0.0)

    def test_loss_rate_calculated(self):
        s = FilterStats(name="x", total=10, kept=3, rejected=7)
        self.assertEqual(s.loss_rate, 0.7)

    def test_to_dict_has_loss_rate(self):
        s = FilterStats(name="x", total=10, kept=3, rejected=7,
                        activated=True, activation_reason="testing")
        d = s.to_dict()
        self.assertEqual(d["name"], "x")
        self.assertEqual(d["loss_rate"], 0.7)
        self.assertTrue(d["activated"])


class TestNormalize(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(normalize_for_match("FooBar"), "foobar")

    def test_diacritics_stripped(self):
        # German umlauts → ae/oe/ue, ß → ss
        # other diacritics (French etc.) are simply stripped
        self.assertEqual(normalize_for_match("Müller"), "mueller")
        self.assertEqual(normalize_for_match("Schöne grüße"), "schoene gruesse")
        self.assertEqual(normalize_for_match("Straße"), "strasse")
        self.assertEqual(normalize_for_match("café"), "cafe")

    def test_whitespace_normalized(self):
        self.assertEqual(normalize_for_match("foo  \n bar"), "foo bar")


if __name__ == "__main__":
    unittest.main()
