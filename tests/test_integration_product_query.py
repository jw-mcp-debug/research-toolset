"""
Integration test: a product-comparison query end to end through the
classifier + filter stack.

The scenario:
  - query: "Vergleich DGX B300 vs B200 — Preis-Leistungs-Verhältnis"
  - a capitalisation heuristic would extract 'Preis' / 'Leistungs' /
    'Verhältnis' as "person components"
  - the PersonHallucinationFilter would then discard all 528 extracts from
    78 sources
  - result: an empty report despite plenty of data

This test runs the pipeline:
  1. classify_query_anchor with the MockLLM, which returns a realistic
     classifier result
  2. PersonHallucinationFilter.applies_to with the returned anchor
  3. run_filter_pipeline on 528 simulated extracts

The protection lies here:
  - the `query_anchor` classifier with the confidence threshold 0.7 in is_person()
  - PersonHallucinationFilter.applies_to checks is_person()
  - if the classifier correctly says "no person" → filter inactive
  - if it is unsure (confidence < 0.7) → inactive as well

Expectation: filter inactive → all extracts survive → the report would not
be empty.

This complements the unit tests with an integration view close to what
happens in the orchestrator (without booting the orchestrator).
"""

import unittest

from src.pipeline.classifiers.query_anchor import classify_query_anchor
from src.pipeline.filters.base import run_filter_pipeline
from src.pipeline.filters.person_match import PersonHallucinationFilter
from tests._helpers import MockLLM, async_test


class _Ctx:
    def __init__(self, anchor):
        self.query_anchor = anchor


class TestProductQueryIntegration(unittest.TestCase):

    @async_test
    async def test_dgx_full_pipeline_no_extracts_lost(self):
        """End to end: product query → 528 extracts survive the filter pipeline.

        Runs through the WHOLE classifier + filter layer, the way the
        orchestrator calls it.
        """
        # ── Step 1: classify_query_anchor ──
        # realistic LLM result for the product query
        anchor_llm = MockLLM().queue_json({
            "anchor_type": "none",
            "target": "",
            "confidence": 0.94,
            "reasoning": (
                "Vergleichsanfrage zu Hardware-Spezifikationen. Die "
                "Query nennt zwei Produkte (DGX B300, DGX B200) und ein "
                "Bewertungs-Kriterium (Preis-Leistungs-Verhältnis). "
                "Keine Person und keine spezifische Organisation."
            ),
        })

        anchor, classifier_call = await classify_query_anchor(
            query="Vergleich DGX B300 vs B200 — Preis-Leistungs-Verhältnis",
            plan_summary=(
                "Strukturierter Vergleich der NVIDIA DGX-Modelle B300 "
                "und B200 hinsichtlich Performance, Stromaufnahme und "
                "Preispunkt."
            ),
            llm=anchor_llm,
        )

        # the classifier recognised correctly: no person
        self.assertEqual(anchor.type, "none")
        self.assertFalse(anchor.is_person())

        # ── Step 2: filter activation via is_person() ──
        # The only protection: anchor.is_person() is False, so the filter is
        # not activated. The confidence threshold 0.7 in is_person() is the
        # second line of defence — even if the classifier says
        # type="person" but confidence < 0.7, the filter does not run.
        person_filter_active = anchor.is_person()
        self.assertFalse(person_filter_active)

        # ── Step 3: filter pipeline ──
        # 528 extracts from 26 sources — realistic figures
        sources_by_url = {
            f"https://nvidia.com/dgx-{i}":
                f"DGX B300 vs B200 specifications: ... (Quelle {i})"
            for i in range(26)
        }
        extracts = []
        for src_idx in range(26):
            for ex_idx in range(528 // 26 + 1):
                extracts.append({
                    "source_url": f"https://nvidia.com/dgx-{src_idx}",
                    "fact": f"Spec-Detail {src_idx}-{ex_idx}",
                    "question_id": "F1",
                })
        extracts = extracts[:528]
        self.assertEqual(len(extracts), 528)

        ctx = _Ctx(anchor=anchor if person_filter_active else None)
        person_filter = PersonHallucinationFilter(
            source_content_lookup=sources_by_url.get,
        )

        kept, stats_map = await run_filter_pipeline(
            items=extracts, ctx=ctx, filters=[person_filter],
        )

        # ── Step 4: verification ──
        # CORE acceptance: all 528 extracts are kept
        self.assertEqual(len(kept), 528,
                         "product query: extracts lost!")
        self.assertFalse(stats_map["person_hallucination"].activated)
        self.assertEqual(stats_map["person_hallucination"].rejected, 0)

    @async_test
    async def test_low_confidence_hallucination_blocked(self):
        """Defence in depth: with low confidence the filter does NOT run.

        If the unsure classifier wrongly says "person" (confidence < 0.7),
        is_person() does not activate — so the filter does not run.
        """
        # the LLM returns "person" with low confidence
        bad_llm = MockLLM().queue_json({
            "anchor_type": "person",
            "target": "Preis Leistungs Verhältnis",
            "confidence": 0.55,  # below the threshold 0.7
            "reasoning": "Unsichere Klassifikation, vermutlich Fehler.",
        })

        anchor, _ = await classify_query_anchor(
            query="Vergleich DGX B300 vs B200 — Preis-Leistungs-Verhältnis",
            plan_summary="",
            llm=bad_llm,
        )

        # With confidence < 0.7 the classifier forces type → "none"
        # (safe default in classify_query_anchor itself).
        # is_person() is thus False in any case.
        self.assertEqual(anchor.type, "none")
        self.assertFalse(
            anchor.is_person(),
            "the confidence threshold 0.7 does not protect — extracts at risk!",
        )


if __name__ == "__main__":
    unittest.main()
