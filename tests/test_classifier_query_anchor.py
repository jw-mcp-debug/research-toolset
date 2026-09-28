"""
Tests for `src.pipeline.classifiers.query_anchor`.

CRITICAL: a heuristic on capitalised nouns misreads German subject-matter
requests as person searches; these tests make sure the classifier does not.

Acceptance:
  - query "DGX B300 vs B200 Preis-Leistungs-Verhältnis" → anchor_type="none"
  - query "Jonas Brenner Forschung" → anchor_type="person", confidence > 0.8
  - query "Lebenslauf Dr. Schmidt Example University" → anchor_type="person",
    target contains "Schmidt"
  - query "KI-Regulierung in der EU" → anchor_type="none"
  - with confidence < 0.7 the type is forced to "none"

The tests use MockLLM. We simulate what a well-functioning LLM WOULD
answer to these queries.
"""

import unittest

from src.pipeline.classifiers.query_anchor import (
    QueryAnchor,
    classify_query_anchor,
)
from tests._helpers import (
    MockLLM,
    assert_classifier_result_consistency,
    async_test,
)


class TestQueryAnchor(unittest.TestCase):

    @async_test
    async def test_dgx_query_returns_none(self):
        """A product-comparison query must NOT have a person anchor.

        A capitalisation heuristic would set `target_person` to
        ['Preis', 'Leistungs', 'Verhältnis'] — this test shows that the
        classifier answers "none".
        """
        llm = MockLLM().queue_json({
            "anchor_type": "none",
            "target": "",
            "confidence": 0.94,
            "reasoning": (
                "Vergleichsanfrage zu Hardware-Spezifikationen. Die "
                "Query nennt zwei Produkte (DGX B300, DGX B200) und "
                "ein Bewertungs-Kriterium (Preis-Leistungs-Verhältnis). "
                "Keine Person und keine spezifische Organisation als "
                "Untersuchungsgegenstand."
            ),
        })

        anchor, call = await classify_query_anchor(
            query="Vergleich DGX B300 vs B200 — Preis-Leistungs-Verhältnis",
            plan_summary="Vergleich der Hardware-Produkte DGX B300 und B200",
            llm=llm,
        )

        self.assertEqual(anchor.type, "none")
        self.assertEqual(anchor.target, "")
        self.assertGreater(anchor.confidence, 0.9)
        self.assertFalse(anchor.is_person())
        self.assertFalse(anchor.fallback_used)
        # consistency: high confidence → reasoning > 50 characters
        assert_classifier_result_consistency(self, anchor)
        # logging: prompt_hash + duration are set
        self.assertEqual(call.name, "query_anchor")
        self.assertNotEqual(call.prompt_hash, "")
        self.assertGreater(call.duration_seconds, 0.0)

    @async_test
    async def test_named_person_returns_person(self):
        """'Jonas Brenner Forschung' = person anchor."""
        llm = MockLLM().queue_json({
            "anchor_type": "person",
            "target": "Jonas Brenner",
            "confidence": 0.91,
            "reasoning": (
                "Die Query nennt einen Vor- und Nachnamen, gefolgt vom "
                "thematischen Zusatz 'Forschung'. Klares Personen-"
                "Recherche-Muster: Fakten ÜBER eine spezifische Person "
                "sammeln."
            ),
        })

        anchor, _ = await classify_query_anchor(
            query="Jonas Brenner Forschung",
            plan_summary="",
            llm=llm,
        )

        self.assertEqual(anchor.type, "person")
        self.assertEqual(anchor.target, "Jonas Brenner")
        self.assertGreater(anchor.confidence, 0.8)
        self.assertTrue(anchor.is_person())
        assert_classifier_result_consistency(self, anchor)

    @async_test
    async def test_dr_schmidt_target_contains_schmidt(self):
        """A CV request recognises 'Schmidt'."""
        llm = MockLLM().queue_json({
            "anchor_type": "person",
            "target": "Dr. Schmidt",
            "confidence": 0.88,
            "reasoning": (
                "Lebenslauf ist explizites biographisches Recherche-"
                "Muster. 'Dr. Schmidt' ist klarer Personen-Bezug, 'Example "
                "University' ist nur die institutionelle Verortung."
            ),
        })

        anchor, _ = await classify_query_anchor(
            query="Lebenslauf Dr. Schmidt Example University",
            plan_summary="",
            llm=llm,
        )

        self.assertEqual(anchor.type, "person")
        self.assertIn("Schmidt", anchor.target)
        self.assertTrue(anchor.is_person())

    @async_test
    async def test_eu_regulation_returns_none(self):
        """'KI-Regulierung in der EU' = no anchor."""
        llm = MockLLM().queue_json({
            "anchor_type": "none",
            "target": "",
            "confidence": 0.96,
            "reasoning": (
                "Sachthema mit geographischem Geltungsbereich (EU). EU "
                "ist hier kein Untersuchungsobjekt sondern Geltungsraum."
            ),
        })

        anchor, _ = await classify_query_anchor(
            query="Aktuelle KI-Regulierung in der EU",
            plan_summary="",
            llm=llm,
        )

        self.assertEqual(anchor.type, "none")
        self.assertFalse(anchor.is_person())
        self.assertFalse(anchor.is_organization())

    @async_test
    async def test_low_confidence_forces_none(self):
        """With confidence < 0.7, type='none' is forced.

        This is the central safe default — better no filter than a wrongly
        activated filter that discards all extracts.
        """
        # The LLM is unsure and says "person" with low confidence
        llm = MockLLM().queue_json({
            "anchor_type": "person",
            "target": "Möglicherweise jemand",
            "confidence": 0.55,
            "reasoning": "Unklare Anfrage, könnte Person sein oder auch nicht",
        })

        anchor, _ = await classify_query_anchor(
            query="Was ist mit denen los",
            plan_summary="",
            llm=llm,
        )

        # despite the LLM answer "person": forced to "none"
        self.assertEqual(anchor.type, "none")
        self.assertEqual(anchor.target, "")
        self.assertTrue(anchor.fallback_used)
        self.assertIn("Confidence", anchor.reasoning)

    @async_test
    async def test_org_anchor_with_high_confidence(self):
        """Organisation anchors are recognised too."""
        llm = MockLLM().queue_json({
            "anchor_type": "organization",
            "target": "Charité Berlin",
            "confidence": 0.93,
            "reasoning": (
                "Anfrage zur Geschichte einer spezifischen Institution. "
                "Charité Berlin ist klar als Untersuchungsgegenstand "
                "benannt."
            ),
        })

        anchor, _ = await classify_query_anchor(
            query="Geschichte der Charité Berlin",
            plan_summary="",
            llm=llm,
        )

        self.assertEqual(anchor.type, "organization")
        self.assertEqual(anchor.target, "Charité Berlin")
        self.assertTrue(anchor.is_organization())
        self.assertFalse(anchor.is_person())

    @async_test
    async def test_invalid_anchor_type_falls_back_to_none(self):
        """Schema violation in the LLM output: fall back to 'none'."""
        llm = MockLLM().queue_json({
            "anchor_type": "tier",  # not in {person, organization, none}
            "target": "Bello",
            "confidence": 0.9,
            "reasoning": "Hund",
        })

        anchor, _ = await classify_query_anchor(
            query="...", plan_summary="", llm=llm,
        )

        self.assertEqual(anchor.type, "none")

    @async_test
    async def test_person_without_target_falls_back(self):
        """Consistency: type='person' but empty target → forced to 'none'."""
        llm = MockLLM().queue_json({
            "anchor_type": "person",
            "target": "",
            "confidence": 0.85,
            "reasoning": "Person, aber Name nicht klar",
        })

        anchor, _ = await classify_query_anchor(
            query="...", plan_summary="", llm=llm,
        )

        self.assertEqual(anchor.type, "none")

    @async_test
    async def test_llm_failure_falls_back_to_none(self):
        """The LLM raises an exception → conservative default 'none'.

        On an LLM timeout: "let everything through" rather than "filter
        everything". Concretely: no person anchor, no filter.
        """
        llm = MockLLM()
        llm.raise_on_call = RuntimeError("network down")

        anchor, call = await classify_query_anchor(
            query="...", plan_summary="", llm=llm,
        )

        self.assertEqual(anchor.type, "none")
        self.assertTrue(anchor.fallback_used)
        self.assertTrue(call.fallback_used)
        self.assertIn("network down", call.fallback_reason)

    @async_test
    async def test_unparseable_response_falls_back(self):
        """JSON parse error → conservative default 'none'."""
        llm = MockLLM().queue("Tut mir leid, kann ich nicht beantworten.")
        # the retry happens automatically, but the queue is empty then
        # → default_response = "" → no JSON either

        anchor, call = await classify_query_anchor(
            query="...", plan_summary="", llm=llm,
        )

        self.assertEqual(anchor.type, "none")
        self.assertTrue(anchor.fallback_used)
        self.assertTrue(call.fallback_used)

    @async_test
    async def test_markdown_wrapped_response_parsed(self):
        """An LLM answer in a Markdown code block is parsed cleanly."""
        llm = MockLLM().queue_markdown_json({
            "anchor_type": "none",
            "target": "",
            "confidence": 0.9,
            "reasoning": "Ist ein längerer Reasoning-Text der sicher fünfzig Zeichen überschreitet",
        })

        anchor, _ = await classify_query_anchor(
            query="DGX B300 specs", plan_summary="", llm=llm,
        )

        self.assertEqual(anchor.type, "none")
        self.assertGreater(anchor.confidence, 0.85)


class TestQueryAnchorDataclass(unittest.TestCase):
    """Direct tests of the QueryAnchor data class (without an LLM)."""

    def test_is_person_requires_threshold(self):
        a = QueryAnchor(type="person", target="X Y", confidence=0.69)
        self.assertFalse(a.is_person())  # 0.69 < 0.7

        a = QueryAnchor(type="person", target="X Y", confidence=0.70)
        self.assertTrue(a.is_person())   # exactly 0.7

    def test_is_person_requires_target(self):
        a = QueryAnchor(type="person", target="", confidence=0.99)
        self.assertFalse(a.is_person())

    def test_default_is_none(self):
        a = QueryAnchor()
        self.assertEqual(a.type, "none")
        self.assertEqual(a.target, "")
        self.assertEqual(a.confidence, 0.0)
        self.assertFalse(a.is_person())


if __name__ == "__main__":
    unittest.main()
