"""
Tests for batched classifiers:
  - source_relevance
  - factoids
"""

import unittest

from src.pipeline.classifiers.factoids import (
    Factoid,
    extract_factoids,
    verify_factoids_against_extracts,
)
from src.pipeline.classifiers.source_relevance import (
    SourceRelevance,
    judge_source_relevance,
)
from tests._helpers import MockLLM, async_test


# ═══════════════════════════════════════════════════════════════════
# judge_source_relevance (batched, 5 per call)
# ═══════════════════════════════════════════════════════════════════


class TestSourceRelevance(unittest.TestCase):

    @async_test
    async def test_basic_batch(self):
        """5 items in one batch — 1 LLM call."""
        questions = [
            {"id": "F1", "question": "Stromaufnahme der DGX B300?"},
        ]
        results = [
            {"title": "NVIDIA DGX B300 Datasheet (PDF)",
             "url": "https://nvidia.com/dgx-b300/datasheet.pdf",
             "snippet": "Power: 14kW typical, 16kW max. ..."},
            {"title": "ServeTheHome: DGX B300 Review",
             "url": "https://servethehome.com/dgx-b300",
             "snippet": "Detailed power consumption testing..."},
            {"title": "Tag-Sammelseite: NVIDIA",
             "url": "https://example.com/tags/nvidia",
             "snippet": "Alle Artikel zu NVIDIA: AI, Gaming, Crypto, Cars..."},
            {"title": "Ein NVIDIA-Aktien-Newsticker",
             "url": "https://news.example.com/stocks/nvda",
             "snippet": "Aktienkurs heute: $1234, Q3-Earnings..."},
            {"title": "AnandTech: B300 vs MI300",
             "url": "https://anandtech.com/b300-vs-mi300",
             "snippet": "Comparison of accelerators..."},
        ]

        llm = MockLLM().queue_json([
            {"index": 1, "relevance": "high", "confidence": 0.95,
             "reasoning": "Offizielles Datasheet, fokussiert auf Frage"},
            {"index": 2, "relevance": "high", "confidence": 0.9,
             "reasoning": "Detailliertes Review zum Thema"},
            {"index": 3, "relevance": "off_topic", "confidence": 0.85,
             "reasoning": "Sammelseite mit gemischten Themen"},
            {"index": 4, "relevance": "off_topic", "confidence": 0.9,
             "reasoning": "Aktien-News, nicht Hardware-Spec"},
            {"index": 5, "relevance": "medium", "confidence": 0.7,
             "reasoning": "Vergleich mit anderem Produkt, könnte Detail haben"},
        ])

        judgments, calls = await judge_source_relevance(
            questions=questions, search_results=results, llm=llm,
        )

        # Exactly 5 verdicts, in input order
        self.assertEqual(len(judgments), 5)
        self.assertEqual(judgments[0].relevance, "high")
        self.assertEqual(judgments[2].relevance, "off_topic")
        self.assertEqual(judgments[4].relevance, "medium")

        # Exactly 1 batch (5 items, batch_size=5)
        self.assertEqual(len(calls), 1)

    @async_test
    async def test_batching_across_multiple_calls(self):
        """12 items with batch_size=5 → 3 batches."""
        questions = [{"id": "F1", "question": "..."}]
        results = [
            {"title": f"R{i}", "url": f"https://x/{i}", "snippet": "..."}
            for i in range(12)
        ]

        llm = MockLLM()
        # 3 Batches: 5, 5, 2 Items
        for batch_size in [5, 5, 2]:
            llm.queue_json([
                {"index": i + 1, "relevance": "medium", "confidence": 0.6,
                 "reasoning": "x"}
                for i in range(batch_size)
            ])

        judgments, calls = await judge_source_relevance(
            questions=questions, search_results=results, llm=llm,
            batch_size=5,
        )

        self.assertEqual(len(judgments), 12)
        self.assertEqual(len(calls), 3)

    @async_test
    async def test_should_fetch_high_confidence_off_topic_filtered(self):
        """should_fetch() is False only for off_topic with confidence > 0.7."""
        # off_topic with high confidence: do not fetch
        j1 = SourceRelevance(relevance="off_topic", confidence=0.9, index=0)
        self.assertFalse(j1.should_fetch())

        # off_topic with low confidence: fetch anyway (safety)
        j2 = SourceRelevance(relevance="off_topic", confidence=0.5, index=1)
        self.assertTrue(j2.should_fetch())

        # everything else: fetch
        for rel in ["high", "medium", "low"]:
            j = SourceRelevance(relevance=rel, confidence=0.95, index=0)
            self.assertTrue(j.should_fetch(),
                            f"{rel} should be fetchable")

    @async_test
    async def test_llm_failure_falls_back_to_medium(self):
        """LLM error → all 'medium' (fetch to be safe)."""
        questions = [{"id": "F1", "question": "..."}]
        results = [{"title": "x", "url": "https://x", "snippet": "..."}]

        llm = MockLLM()
        llm.raise_on_call = RuntimeError("down")

        judgments, calls = await judge_source_relevance(
            questions=questions, search_results=results, llm=llm,
        )

        self.assertEqual(len(judgments), 1)
        self.assertEqual(judgments[0].relevance, "medium")
        self.assertTrue(judgments[0].fallback_used)
        self.assertTrue(judgments[0].should_fetch())

    @async_test
    async def test_partial_response_padded_with_fallback(self):
        """The LLM answers with fewer items than asked → the rest = medium default."""
        questions = [{"id": "F1", "question": "..."}]
        results = [
            {"title": f"R{i}", "url": f"https://x/{i}", "snippet": "..."}
            for i in range(3)
        ]

        # The LLM returns only 2 items (not 3)
        llm = MockLLM().queue_json([
            {"index": 1, "relevance": "high", "confidence": 0.9,
             "reasoning": "x"},
            {"index": 2, "relevance": "medium", "confidence": 0.6,
             "reasoning": "x"},
        ])

        judgments, _ = await judge_source_relevance(
            questions=questions, search_results=results, llm=llm,
        )

        self.assertEqual(len(judgments), 3)
        self.assertEqual(judgments[0].relevance, "high")
        self.assertEqual(judgments[1].relevance, "medium")
        # Item 3 = Default
        self.assertEqual(judgments[2].relevance, "medium")

    @async_test
    async def test_empty_input(self):
        judgments, calls = await judge_source_relevance(
            questions=[], search_results=[], llm=MockLLM(),
        )
        self.assertEqual(judgments, [])
        self.assertEqual(calls, [])


# ═══════════════════════════════════════════════════════════════════
# extract_factoids
# ═══════════════════════════════════════════════════════════════════


class TestExtractFactoids(unittest.TestCase):

    @async_test
    async def test_extract_typed_factoids(self):
        report = """
        Die DGX B300 verbraucht typisch 14kW und kostet ~500.000 USD.
        Veröffentlichung: 17.10.2024. Vergleichstest in DOI 10.1145/12345.
        """
        llm = MockLLM().queue_json([
            {"factoid": "Die DGX B300 verbraucht typisch 14kW",
             "type": "numeric_spec",
             "report_position": "Erste Zeile"},
            {"factoid": "Kosten ~500.000 USD",
             "type": "numeric_spec",
             "report_position": "Erste Zeile"},
            {"factoid": "Veröffentlichung: 17.10.2024",
             "type": "date",
             "report_position": "Zweite Zeile"},
            {"factoid": "DOI 10.1145/12345",
             "type": "identifier",
             "report_position": "Zweite Zeile"},
        ])

        factoids, call = await extract_factoids(report, llm=llm)

        self.assertEqual(len(factoids), 4)
        types = {f.type for f in factoids}
        self.assertIn("numeric_spec", types)
        self.assertIn("date", types)
        self.assertIn("identifier", types)
        self.assertEqual(call.output["n_factoids"], 4)

    @async_test
    async def test_invalid_type_falls_back_to_claim(self):
        llm = MockLLM().queue_json([
            {"factoid": "Etwas",
             "type": "unbekannter_typ",  # not in VALID_FACTOID_TYPES
             "report_position": "..."},
        ])

        factoids, _ = await extract_factoids("test", llm=llm)
        self.assertEqual(len(factoids), 1)
        self.assertEqual(factoids[0].type, "claim")

    @async_test
    async def test_empty_factoid_skipped(self):
        llm = MockLLM().queue_json([
            {"factoid": "Echtes Faktoid", "type": "claim"},
            {"factoid": "", "type": "claim"},
            {"factoid": "   ", "type": "claim"},
        ])

        factoids, _ = await extract_factoids("test", llm=llm)
        self.assertEqual(len(factoids), 1)
        self.assertEqual(factoids[0].factoid, "Echtes Faktoid")

    @async_test
    async def test_llm_failure_returns_empty(self):
        llm = MockLLM()
        llm.raise_on_call = RuntimeError("x")

        factoids, call = await extract_factoids("test", llm=llm)

        self.assertEqual(factoids, [])
        self.assertTrue(call.fallback_used)


# ═══════════════════════════════════════════════════════════════════
# verify_factoids_against_extracts
# ═══════════════════════════════════════════════════════════════════


class TestVerifyFactoids(unittest.TestCase):

    @async_test
    async def test_verified_supported_factoid(self):
        factoids = [
            Factoid(factoid="DGX B300 verbraucht 14kW", type="numeric_spec"),
        ]
        extracts = [
            {"id": "E1", "fact": "Power: 14kW typ. (NVIDIA Datasheet)",
             "source_title": "NVIDIA Datasheet"},
        ]

        llm = MockLLM().queue_json([
            {"factoid_index": 1, "verified": "true", "confidence": 0.92,
             "supporting_extract_id": "E1",
             "supporting_quote": "Power: 14kW typ.",
             "reasoning": "Exakte Übereinstimmung mit Primärquelle"},
        ])

        verifications, calls = await verify_factoids_against_extracts(
            factoids, extracts, llm=llm,
        )

        self.assertEqual(len(verifications), 1)
        self.assertEqual(verifications[0].verified, "true")
        self.assertTrue(verifications[0].is_verified())
        self.assertEqual(verifications[0].supporting_extract_id, "E1")

    @async_test
    async def test_low_confidence_forces_uncertain(self):
        """With conf < 0.5 → 'uncertain'."""
        factoids = [Factoid(factoid="X", type="claim")]
        extracts = [{"id": "E1", "fact": "Y", "source_title": "..."}]

        llm = MockLLM().queue_json([
            {"factoid_index": 1, "verified": "true", "confidence": 0.3,
             "supporting_extract_id": "E1",
             "reasoning": "Bin mir nicht sicher"},
        ])

        verifications, _ = await verify_factoids_against_extracts(
            factoids, extracts, llm=llm,
        )

        self.assertEqual(verifications[0].verified, "uncertain")

    @async_test
    async def test_unverified_high_confidence_marker(self):
        """A factoid that is clearly wrong → is_unverified() True for marking."""
        factoids = [
            Factoid(factoid="Erschienen 1999", type="date"),
        ]
        extracts = [
            {"id": "E1", "fact": "Erschienen 17.10.2024",
             "source_title": "Datasheet"},
        ]

        llm = MockLLM().queue_json([
            {"factoid_index": 1, "verified": "false", "confidence": 0.95,
             "supporting_extract_id": "E1",
             "reasoning": "Datum widerspricht der Quelle (2024 vs 1999)"},
        ])

        verifications, _ = await verify_factoids_against_extracts(
            factoids, extracts, llm=llm,
        )

        self.assertEqual(verifications[0].verified, "false")
        self.assertTrue(verifications[0].is_unverified())  # → marker in the report

    @async_test
    async def test_batching(self):
        """7 factoids with batch_size=5 → 2 batches."""
        factoids = [
            Factoid(factoid=f"F{i}", type="claim") for i in range(7)
        ]
        extracts = [{"id": f"E{i}", "fact": "x", "source_title": "y"}
                    for i in range(3)]

        llm = MockLLM()
        # First batch: 5 items
        llm.queue_json([
            {"factoid_index": i + 1, "verified": "uncertain",
             "confidence": 0.6, "reasoning": "x"}
            for i in range(5)
        ])
        # Second batch: 2 items
        llm.queue_json([
            {"factoid_index": i + 6, "verified": "uncertain",
             "confidence": 0.6, "reasoning": "x"}
            for i in range(2)
        ])

        verifications, calls = await verify_factoids_against_extracts(
            factoids, extracts, llm=llm, batch_size=5,
        )

        self.assertEqual(len(verifications), 7)
        self.assertEqual(len(calls), 2)
        # The index is consistent across batches
        self.assertEqual(verifications[0].factoid_index, 1)
        self.assertEqual(verifications[6].factoid_index, 7)


if __name__ == "__main__":
    unittest.main()
