"""
Tests for `src.pipeline.report_quality`.

Acceptance:
  - check_report_quality returns a clean result for good/deficient reports
  - long report: segmented check
  - findings are merged across segments
  - hallucinations are marked as 'hallucinated'
  - on LLM failure: conservatively fulfilled=True, fallback_used=True
  - check_query_fulfillment works at the level of the user query
  - excerpt for a report that is too long (head + tail)
"""

import unittest

from src.pipeline.models import (
    ResearchPlan,
    ResearchQuestion,
    SourceExtract,
)
from src.pipeline.report_quality import (
    FULFILLMENT_REPORT_MAX,
    FulfillmentResult,
    QualityIssue,
    QUALITY_FULLTEXT_THRESHOLD,
    check_query_fulfillment,
    check_report_quality,
    dedupe_maengel,
    split_for_quality,
)
from tests._helpers import MockLLM, async_test


def _patch_adapter_with(llm: MockLLM):
    from src.llm import classifier_adapter

    class _Pass:
        def __init__(self, _dual): pass
        async def complete(self, messages, max_tokens=None):
            return await llm.complete(messages, max_tokens=max_tokens)

    original = classifier_adapter.HarvestModelAdapter
    classifier_adapter.HarvestModelAdapter = _Pass
    return original


def _restore_adapter(original):
    from src.llm import classifier_adapter
    classifier_adapter.HarvestModelAdapter = original


def _make_plan() -> ResearchPlan:
    return ResearchPlan(
        summary="Recherche zu DGX",
        questions=[
            ResearchQuestion(id="F1", question="Was kostet die DGX?",
                             search_terms=["x"], priority="hoch"),
            ResearchQuestion(id="F2", question="Stromverbrauch?",
                             search_terms=["x"], priority="hoch"),
        ],
    )


# ───────────────────────────────────────────────────────────────────
# split_for_quality
# ───────────────────────────────────────────────────────────────────


class TestSplitForQuality(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(split_for_quality(""), [])

    def test_short_text_one_or_two_segments(self):
        # With little text and no headings: paragraph split
        text = "Kurzer Text mit zwei\n\nAbsätzen."
        segs = split_for_quality(text)
        self.assertGreaterEqual(len(segs), 1)
        # content is preserved
        self.assertEqual(
            "".join(segs).replace("\n\n", "").replace("\n", ""),
            text.replace("\n\n", "").replace("\n", ""),
        )

    def test_heading_based_split_with_enough_headings(self):
        text = (
            "## Einleitung\n\nText 1.\n\n"
            "## Hauptteil\n\nText 2.\n\n"
            "## Schluss\n\nText 3.\n\n"
        )
        segs = split_for_quality(text)
        # 2-3 segments expected with this threshold
        self.assertGreaterEqual(len(segs), 2)
        # headings are contained in the segments
        all_text = "\n".join(segs)
        self.assertIn("Einleitung", all_text)
        self.assertIn("Hauptteil", all_text)
        self.assertIn("Schluss", all_text)


# ───────────────────────────────────────────────────────────────────
# dedupe_maengel
# ───────────────────────────────────────────────────────────────────


class TestDedupeMaengel(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(dedupe_maengel([]), [])

    def test_unique_maengel_pass_through(self):
        issues = [
            {"description": "M1", "kind": "missing", "segment": 1},
            {"description": "M2", "kind": "redundant", "segment": 2},
        ]
        out = dedupe_maengel(issues)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].segments, [1])
        self.assertEqual(out[1].segments, [2])

    def test_duplicate_descriptions_merge(self):
        issues = [
            {"description": "Gleicher Mangel", "kind": "missing", "segment": 1},
            {"description": "Gleicher Mangel", "kind": "missing", "segment": 2},
            {"description": "Gleicher Mangel", "kind": "missing", "segment": 3},
        ]
        out = dedupe_maengel(issues)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].segments, [1, 2, 3])

    def test_case_insensitive_match(self):
        issues = [
            {"description": "Halluzination bei Preisangabe", "segment": 1},
            {"description": "HALLUZINATION BEI PREISANGABE", "segment": 2},
        ]
        out = dedupe_maengel(issues)
        self.assertEqual(len(out), 1)
        self.assertEqual(sorted(out[0].segments), [1, 2])

    def test_string_maengel_kept(self):
        issues = ["Roh-Strings ohne Dict-Struktur"]
        out = dedupe_maengel(issues)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].description, "Roh-Strings ohne Dict-Struktur")

    def test_empty_descriptions_skipped(self):
        out = dedupe_maengel([
            {"description": "", "kind": "missing"},
            {"description": "  ", "kind": "missing"},
        ])
        self.assertEqual(out, [])


# ───────────────────────────────────────────────────────────────────
# check_report_quality — full text
# ───────────────────────────────────────────────────────────────────


class TestCheckReportQualityFulltext(unittest.TestCase):

    @async_test
    async def test_clean_report_passes(self):
        llm = MockLLM().queue_json({
            "passed": True, "rating": "good", "issues": [],
        })
        original = _patch_adapter_with(llm)
        try:
            result = await check_report_quality(
                report="## Bericht\n\nKurz aber gut.",
                extracts=[
                    SourceExtract("u", "t", "F1", "Fakt",
                                  polarity="positive"),
                ],
                plan=_make_plan(),
                dual_llm=None,
            )
        finally:
            _restore_adapter(original)

        self.assertTrue(result.passed)
        self.assertEqual(result.rating, "good")
        self.assertEqual(result.issues, [])
        self.assertEqual(result.segments, 1)
        self.assertFalse(result.fallback_used)

    @async_test
    async def test_hallucination_detected(self):
        llm = MockLLM().queue_json({
            "passed": False,
            "rating": "poor",
            "issues": [{
                "question_id": "F1",
                "kind": "hallucinated",
                "description": "Preis 500$ steht nicht in den Extrakten",
                "suggestion": "Mit Quelle belegen oder weglassen",
            }],
        })
        original = _patch_adapter_with(llm)
        try:
            result = await check_report_quality(
                report="## Bericht\n\nDie DGX kostet 500$.",
                extracts=[
                    SourceExtract("u", "t", "F1", "Andere Aussage",
                                  polarity="positive"),
                ],
                plan=_make_plan(),
                dual_llm=None,
            )
        finally:
            _restore_adapter(original)

        self.assertFalse(result.passed)
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].kind, "hallucinated")

    @async_test
    async def test_empty_report_returns_fallback(self):
        result = await check_report_quality(
            report="", extracts=[], plan=None, dual_llm=None,
        )
        self.assertTrue(result.passed)
        self.assertTrue(result.fallback_used)

    @async_test
    async def test_llm_failure_returns_fallback(self):
        from src.llm import classifier_adapter

        class _Broken:
            def __init__(self, _dual): pass
            async def complete(self, messages, max_tokens=None):
                raise RuntimeError("LLM down")

        original = classifier_adapter.HarvestModelAdapter
        classifier_adapter.HarvestModelAdapter = _Broken
        try:
            result = await check_report_quality(
                report="## Bericht",
                extracts=[
                    SourceExtract("u", "t", "F1", "x", polarity="positive"),
                ],
                plan=_make_plan(),
                dual_llm=None,
            )
        finally:
            classifier_adapter.HarvestModelAdapter = original

        self.assertTrue(result.passed)
        self.assertTrue(result.fallback_used)

    @async_test
    async def test_only_positive_extracts_in_prompt(self):
        """Negative/meta extracts do NOT go into the prompt as the factual basis."""
        llm = MockLLM().queue_json({"passed": True, "issues": []})

        captured = {}

        from src.llm import classifier_adapter

        class _Capturing:
            def __init__(self, _dual): pass
            async def complete(self, messages, max_tokens=None):
                captured["messages"] = messages
                return await llm.complete(messages, max_tokens=max_tokens)

        original = classifier_adapter.HarvestModelAdapter
        classifier_adapter.HarvestModelAdapter = _Capturing
        try:
            await check_report_quality(
                report="## Bericht",
                extracts=[
                    SourceExtract("u", "t", "F1", "POSITIV-FAKT",
                                  polarity="positive"),
                    SourceExtract("u", "t", "F1", "NEGATIV-FAKT",
                                  polarity="negative"),
                    SourceExtract("u", "t", "F1", "META-FAKT",
                                  polarity="meta"),
                ],
                plan=_make_plan(),
                dual_llm=None,
            )
        finally:
            classifier_adapter.HarvestModelAdapter = original

        prompt_text = "\n".join(
            m.get("content", "") for m in captured["messages"]
        )
        self.assertIn("POSITIV-FAKT", prompt_text)
        self.assertNotIn("NEGATIV-FAKT", prompt_text)
        self.assertNotIn("META-FAKT", prompt_text)


# ───────────────────────────────────────────────────────────────────
# check_report_quality — segmented
# ───────────────────────────────────────────────────────────────────


class TestCheckReportQualitySegmented(unittest.TestCase):

    @async_test
    async def test_long_report_uses_segmented_path(self):
        # report > QUALITY_FULLTEXT_THRESHOLD with headings
        long_report = (
            "## Einleitung\n\n" + "Text. " * 200
            + "## Hauptteil\n\n" + "Text. " * 800
            + "## Schluss\n\n" + "Text. " * 1500
        )
        # >15k characters required
        self.assertGreater(len(long_report), QUALITY_FULLTEXT_THRESHOLD)

        # three LLM answers (one per segment)
        llm = MockLLM()
        for _ in range(3):
            llm.queue_json({
                "passed": True, "rating": "good",
                "issues": [],
            })

        original = _patch_adapter_with(llm)
        try:
            result = await check_report_quality(
                report=long_report,
                extracts=[
                    SourceExtract("u", "t", "F1", "x", polarity="positive"),
                ],
                plan=_make_plan(),
                dual_llm=None,
            )
        finally:
            _restore_adapter(original)

        # more than 1 segment checked
        self.assertGreaterEqual(result.segments, 2)
        self.assertTrue(result.passed)

    @async_test
    async def test_segments_merge_duplicate_maengel(self):
        """If two segments report the same finding, it is merged."""
        long_report = (
            "## A\n\n" + "Text. " * 1500
            + "## B\n\n" + "Text. " * 1500
            + "## C\n\n" + "Text. " * 1500
        )
        self.assertGreater(len(long_report), QUALITY_FULLTEXT_THRESHOLD)

        # both segments report the same finding
        llm = MockLLM()
        for _ in range(3):
            llm.queue_json({
                "passed": False, "rating": "poor",
                "issues": [{
                    "question_id": "F1", "kind": "redundant",
                    "description": "Selber Punkt mehrfach erwähnt",
                    "suggestion": "Eine Stelle löschen",
                }],
            })

        original = _patch_adapter_with(llm)
        try:
            result = await check_report_quality(
                report=long_report,
                extracts=[
                    SourceExtract("u", "t", "F1", "x", polarity="positive"),
                ],
                plan=_make_plan(),
                dual_llm=None,
            )
        finally:
            _restore_adapter(original)

        # findings were merged — only 1 entry
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].kind, "redundant")
        # but the segment field shows that several segments are affected
        self.assertGreaterEqual(len(result.issues[0].segments), 2)


# ───────────────────────────────────────────────────────────────────
# check_query_fulfillment
# ───────────────────────────────────────────────────────────────────


class TestCheckQueryFulfillment(unittest.TestCase):

    @async_test
    async def test_fulfilled_returns_true(self):
        llm = MockLLM().queue_json({
            "fulfilled": True, "assessment": "Anfrage komplett beantwortet",
            "rework": "",
        })
        original = _patch_adapter_with(llm)
        try:
            result = await check_query_fulfillment(
                query="Was kostet die DGX?",
                report="## DGX-Preis\n\nDie DGX kostet $415k.",
                dual_llm=None,
            )
        finally:
            _restore_adapter(original)

        self.assertTrue(result.fulfilled)
        self.assertIn("beantwortet", result.assessment.lower())

    @async_test
    async def test_not_fulfilled_with_nacharbeit(self):
        llm = MockLLM().queue_json({
            "fulfilled": False,
            "assessment": "Nur Preis genannt, Stromverbrauch fehlt",
            "rework": "Ergänze einen Abschnitt zum TDP",
        })
        original = _patch_adapter_with(llm)
        try:
            result = await check_query_fulfillment(
                query="Preis und Stromverbrauch der DGX?",
                report="## Preis\n\n$415k.",
                dual_llm=None,
            )
        finally:
            _restore_adapter(original)

        self.assertFalse(result.fulfilled)
        self.assertIn("TDP", result.rework)

    @async_test
    async def test_empty_query_returns_fallback(self):
        result = await check_query_fulfillment("", "Bericht", dual_llm=None)
        self.assertTrue(result.fulfilled)
        self.assertTrue(result.fallback_used)

    @async_test
    async def test_long_report_uses_excerpt(self):
        """For a very long report: head + [...] + tail."""
        long_report = "X" * (FULFILLMENT_REPORT_MAX * 2)
        captured = {}

        from src.llm import classifier_adapter
        from tests._helpers import MockLLM as _MockLLM
        sub = _MockLLM().queue_json({"fulfilled": True, "assessment": "ok",
                                     "rework": ""})

        class _Capturing:
            def __init__(self, _dual): pass
            async def complete(self, messages, max_tokens=None):
                captured["messages"] = messages
                return await sub.complete(messages, max_tokens=max_tokens)

        original = classifier_adapter.HarvestModelAdapter
        classifier_adapter.HarvestModelAdapter = _Capturing
        try:
            await check_query_fulfillment(
                query="Test", report=long_report, dual_llm=None,
            )
        finally:
            classifier_adapter.HarvestModelAdapter = original

        prompt = "\n".join(
            m.get("content", "") for m in captured["messages"]
        )
        # excerpt marker
        self.assertIn("[...]", prompt)
        # the prompt itself is smaller than the original text
        self.assertLess(len(prompt), len(long_report))

    @async_test
    async def test_llm_failure_returns_fallback(self):
        from src.llm import classifier_adapter

        class _Broken:
            def __init__(self, _dual): pass
            async def complete(self, messages, max_tokens=None):
                raise RuntimeError("oops")

        original = classifier_adapter.HarvestModelAdapter
        classifier_adapter.HarvestModelAdapter = _Broken
        try:
            result = await check_query_fulfillment(
                query="?", report="Bericht", dual_llm=None,
            )
        finally:
            classifier_adapter.HarvestModelAdapter = original

        self.assertTrue(result.fulfilled)
        self.assertTrue(result.fallback_used)


if __name__ == "__main__":
    unittest.main()


def test_german_answers_are_mapped_to_the_english_contract():
    """A model answering with the former German keys/values still parses."""
    from src.pipeline.report_quality import (
        _english_contract,
    )
    q = _english_contract({"bestanden": False, "gesamtbewertung": "mangelhaft",
                           "maengel": [{"art": "halluziniert", "beschreibung": "x"}]})
    assert q == {"passed": False, "rating": "poor",
                 "issues": [{"kind": "hallucinated", "description": "x"}]}
    assert QualityIssue.from_dict(q["issues"][0]).kind == "hallucinated"
    f = FulfillmentResult.from_dict(_english_contract({"erfuellt": False, "nacharbeit": "y"}))
    assert f.fulfilled is False and f.rework == "y"
