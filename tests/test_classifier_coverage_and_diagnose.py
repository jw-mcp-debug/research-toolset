"""
Tests for `coverage`, `continue_research`, `diagnose_pipeline_state`.

Acceptance:
  - the classifiers use no mechanical thresholds (>= 3 and >= 2,
    == 0 → break, >= 2 → "unanswerable").
  - a run in which a filter discarded everything ends with
    diagnosis="filter_too_strict"
  - real coverage with 3 strong sources is recognised as
    coverage="answered" with confidence > 0.8
  - with low classifier confidence (< 0.5) the decision is conservative
    ("one more round")
"""

import unittest

from src.pipeline.classifiers.coverage import (
    decide_continue_research,
    evaluate_coverage,
)
from src.pipeline.classifiers.diagnose import (
    diagnose_pipeline_state,
)
from tests._helpers import (
    MockLLM,
    assert_classifier_result_consistency,
    async_test,
)


# ═══════════════════════════════════════════════════════════════════
# evaluate_coverage
# ═══════════════════════════════════════════════════════════════════


class TestEvaluateCoverage(unittest.TestCase):

    @async_test
    async def test_answered_with_strong_sources(self):
        """3 strong sources → answered with confidence > 0.8."""
        extracts = [
            {"id": "E1", "fact": "DGX B300 verbraucht 14kW typ.",
             "source_title": "NVIDIA Datasheet", "reliability": "hoch"},
            {"id": "E2", "fact": "DGX B300: TDP 14kW",
             "source_title": "ServeTheHome Review", "reliability": "hoch"},
            {"id": "E3", "fact": "Power-Budget 14kW im Rack",
             "source_title": "AnandTech", "reliability": "hoch"},
        ]
        llm = MockLLM().queue_json({
            "coverage": "answered",
            "confidence": 0.92,
            "missing_aspects": [],
            "supporting_extract_ids": ["E1", "E2", "E3"],
            "reasoning": (
                "Drei unabhängige Quellen, davon eine Primärquelle "
                "(NVIDIA Datasheet) und zwei seriöse Reviews, geben "
                "übereinstimmend 14kW als Stromaufnahme an."
            ),
        })

        result, call = await evaluate_coverage(
            question_id="F1",
            question="Wie hoch ist die Stromaufnahme der DGX B300?",
            priority="hoch",
            extracts=extracts,
            active_filters=["negative_extract"],
            filter_loss_rate=0.05,
            llm=llm,
        )

        self.assertEqual(result.coverage, "answered")
        self.assertGreater(result.confidence, 0.8)
        self.assertEqual(result.missing_aspects, [])
        self.assertIn("E1", result.supporting_extract_ids)
        assert_classifier_result_consistency(self, result)

    @async_test
    async def test_filter_blocked_when_loss_high_and_no_extracts(self):
        """High filter loss rate + no extracts → filter_blocked.

        This is the key behaviour: the LLM SHOULD recognise that it is not
        the world that is empty, but that the filter swallowed everything.
        """
        llm = MockLLM().queue_json({
            "coverage": "filter_blocked",
            "confidence": 0.87,
            "missing_aspects": ["Stromaufnahme", "Kühlung", "Speicher"],
            "supporting_extract_ids": [],
            "reasoning": (
                "Filter-Verlustrate 95%, keine positiven Extrakte trotz "
                "26 besuchter Quellen. Klares Indiz dass der Filter die "
                "Inhalte verworfen hat, nicht dass die Quellen keine "
                "Information enthalten."
            ),
        })

        result, _ = await evaluate_coverage(
            question_id="F1",
            question="Wie hoch ist die Stromaufnahme der DGX B300?",
            priority="hoch",
            extracts=[],
            active_filters=["person_hallucination"],
            filter_loss_rate=0.95,
            llm=llm,
        )

        self.assertEqual(result.coverage, "filter_blocked")
        self.assertGreater(result.confidence, 0.8)

    @async_test
    async def test_low_confidence_forces_partial(self):
        """With confidence < 0.5 → 'partial' as the safe default."""
        llm = MockLLM().queue_json({
            "coverage": "answered",
            "confidence": 0.3,
            "reasoning": "Bin mir nicht sicher",
        })

        result, _ = await evaluate_coverage(
            question_id="F1",
            question="...",
            priority="hoch",
            extracts=[],
            active_filters=[],
            filter_loss_rate=0.0,
            llm=llm,
        )

        self.assertEqual(result.coverage, "partial")
        self.assertTrue(result.fallback_used)

    @async_test
    async def test_unanswered(self):
        llm = MockLLM().queue_json({
            "coverage": "unanswered",
            "confidence": 0.85,
            "missing_aspects": ["alles"],
            "reasoning": (
                "Keine Quelle hat zu dieser Frage etwas beigetragen. "
                "Filter waren nicht aktiv (Verlustrate 0%), die Welt "
                "scheint hier wenig Information zu haben."
            ),
        })

        result, _ = await evaluate_coverage(
            question_id="F1", question="...", priority="hoch",
            extracts=[], active_filters=[], filter_loss_rate=0.0, llm=llm,
        )

        self.assertEqual(result.coverage, "unanswered")
        assert_classifier_result_consistency(self, result)

    @async_test
    async def test_invalid_coverage_falls_back_to_partial(self):
        llm = MockLLM().queue_json({
            "coverage": "vielleicht",  # not in VALID_COVERAGE
            "confidence": 0.8,
            "reasoning": "Schwierig zu sagen",
        })

        result, _ = await evaluate_coverage(
            question_id="F1", question="...", priority="hoch",
            extracts=[], active_filters=[], filter_loss_rate=0.0, llm=llm,
        )

        self.assertEqual(result.coverage, "partial")

    @async_test
    async def test_llm_failure_falls_back_to_partial(self):
        llm = MockLLM()
        llm.raise_on_call = TimeoutError("llm timeout")

        result, call = await evaluate_coverage(
            question_id="F1", question="...", priority="hoch",
            extracts=[], active_filters=[], filter_loss_rate=0.0, llm=llm,
        )

        self.assertEqual(result.coverage, "partial")
        self.assertTrue(result.fallback_used)
        self.assertTrue(call.fallback_used)


# ═══════════════════════════════════════════════════════════════════
# decide_continue_research
# ═══════════════════════════════════════════════════════════════════


class TestDecideContinueResearch(unittest.TestCase):

    @async_test
    async def test_continue_when_questions_open(self):
        llm = MockLLM().queue_json({
            "decision": "continue",
            "confidence": 0.85,
            "next_round_focus": "Fokus auf F3 mit breiteren Suchbegriffen",
            "reasoning": (
                "F1 und F2 beantwortet, F3 partiell. Letzte Runde brachte "
                "noch 12 neue positive Extrakte — diminishing returns "
                "noch nicht erreicht."
            ),
        })

        result, _ = await decide_continue_research(
            round_number=1, max_rounds=3,
            answered_count=2, partial_count=1, unanswered_count=0,
            filter_blocked_count=0, total_count=3,
            new_extracts=12, new_sources=8, filter_loss_rate=0.05,
            total_extracts=45, total_sources=20,
            coverage_details="F1: answered, F2: answered, F3: partial",
            llm=llm,
        )

        self.assertEqual(result.decision, "continue")
        self.assertTrue(result.should_continue())
        self.assertGreater(len(result.next_round_focus), 5)

    @async_test
    async def test_stop_filter_problem_at_high_loss(self):
        """With a high filter loss rate over several rounds → stop_filter_problem."""
        llm = MockLLM().queue_json({
            "decision": "stop_filter_problem",
            "confidence": 0.91,
            "next_round_focus": "",
            "reasoning": (
                "3 von 4 Fragen sind filter_blocked, Filter-Verlustrate "
                "diese Runde 92%. Weitere Runden würden ins Leere "
                "laufen — der Filter ist das Problem."
            ),
        })

        result, _ = await decide_continue_research(
            round_number=2, max_rounds=3,
            answered_count=0, partial_count=1, unanswered_count=0,
            filter_blocked_count=3, total_count=4,
            new_extracts=2, new_sources=15, filter_loss_rate=0.92,
            total_extracts=2, total_sources=30,
            coverage_details="F1: filter_blocked, F2: filter_blocked, F3: filter_blocked, F4: partial",
            llm=llm,
        )

        self.assertEqual(result.decision, "stop_filter_problem")
        self.assertFalse(result.should_continue())

    @async_test
    async def test_low_confidence_forces_continue_when_rounds_left(self):
        """With conf<0.5 and round<max → 'continue' default."""
        llm = MockLLM().queue_json({
            "decision": "stop_done",
            "confidence": 0.3,  # low
            "reasoning": "...",
        })

        result, _ = await decide_continue_research(
            round_number=1, max_rounds=3,
            answered_count=1, partial_count=2, unanswered_count=0,
            filter_blocked_count=0, total_count=3,
            new_extracts=5, new_sources=4, filter_loss_rate=0.1,
            total_extracts=10, total_sources=8,
            coverage_details="...",
            llm=llm,
        )

        self.assertEqual(result.decision, "continue")
        self.assertTrue(result.fallback_used)

    @async_test
    async def test_llm_failure_continues_if_rounds_left(self):
        llm = MockLLM()
        llm.raise_on_call = RuntimeError("down")

        result, _ = await decide_continue_research(
            round_number=0, max_rounds=3,
            answered_count=0, partial_count=0, unanswered_count=3,
            filter_blocked_count=0, total_count=3,
            new_extracts=0, new_sources=0, filter_loss_rate=0.0,
            total_extracts=0, total_sources=0,
            coverage_details="",
            llm=llm,
        )

        self.assertEqual(result.decision, "continue")
        self.assertTrue(result.fallback_used)

    @async_test
    async def test_llm_failure_stops_at_max_rounds(self):
        llm = MockLLM()
        llm.raise_on_call = RuntimeError("down")

        result, _ = await decide_continue_research(
            round_number=3, max_rounds=3,
            answered_count=2, partial_count=1, unanswered_count=0,
            filter_blocked_count=0, total_count=3,
            new_extracts=1, new_sources=1, filter_loss_rate=0.1,
            total_extracts=20, total_sources=15,
            coverage_details="",
            llm=llm,
        )

        # With fallback and round_number >= max_rounds: stop_done
        self.assertEqual(result.decision, "stop_done")


# ═══════════════════════════════════════════════════════════════════
# diagnose_pipeline_state
# ═══════════════════════════════════════════════════════════════════


class TestDiagnose(unittest.TestCase):

    @async_test
    async def test_dgx_reproduction_filter_too_strict(self):
        """CRITICAL: a run in which a filter discarded everything ends with
        filter_too_strict.

        We simulate the final state of such a run:
          - 78 sources visited
          - 528 extracts produced by the harvest
          - all discarded by the person hallucination filter
          - 0 positive extracts at the end
        """
        llm = MockLLM().queue_json({
            "diagnosis": "filter_too_strict",
            "confidence": 0.95,
            "remediation": (
                "Recherche neu starten ohne Personen-Halluzinations-"
                "Filter. Die Anfrage ist eine technische Vergleichs-"
                "anfrage, kein Personen-Bezug."
            ),
            "user_message": (
                "Die Recherche hat 78 Quellen besucht und daraus 528 "
                "Extrakte produziert — aber alle wurden vom Personen-"
                "Halluzinations-Filter verworfen, weil die Anfrage "
                "fälschlich als personenbezogen klassifiziert wurde. "
                "Bitte starten Sie die Recherche ohne diesen Filter neu."
            ),
            "reasoning": (
                "Filter-Verlustrate 100% (528/528 verworfen). "
                "filter_blocked-Anteil 100% der Fragen. Keine plausible "
                "Erklärung außer falscher Filter-Aktivierung."
            ),
        })

        result, _ = await diagnose_pipeline_state(
            query="Vergleich DGX B300 vs B200 — Preis-Leistungs-Verhältnis",
            use_case="web_research",
            n_rounds=2, max_rounds=3,
            stop_reason="stop_filter_problem",
            final_extracts=0,
            positive_extracts=0,
            negative_extracts=0,
            meta_extracts=0,
            n_sources=78,
            coverage_distribution={
                "answered": 0, "partial": 0,
                "unanswered": 0, "filter_blocked": 4,
            },
            filter_stats={
                "person_hallucination": {
                    "activated": True, "rejected": 528, "total": 528,
                },
                "negative_extract": {
                    "activated": True, "rejected": 0, "total": 0,
                },
            },
            anchor_type="person",  # wrongly classified (e.g. because
                                   # the user forced the filter manually)
            anchor_confidence=0.55,
            active_filters=["person_hallucination", "negative_extract"],
            llm=llm,
        )

        # This is the diagnosis that identifies the problem correctly
        self.assertEqual(result.diagnosis, "filter_too_strict")
        self.assertTrue(result.is_problematic())
        self.assertFalse(result.is_successful())
        self.assertGreater(result.confidence, 0.8)
        self.assertIn("Filter", result.user_message)
        self.assertGreater(len(result.remediation), 20)
        assert_classifier_result_consistency(self, result)

    @async_test
    async def test_successful_diagnosis(self):
        llm = MockLLM().queue_json({
            "diagnosis": "successful",
            "confidence": 0.88,
            "remediation": "Keine — Recherche erfolgreich.",
            "user_message": "Recherche abgeschlossen, alle Fragen beantwortet.",
            "reasoning": (
                "Mehrheit answered, niedrige Filter-Verlustrate (5%), "
                "solide Quellenlage."
            ),
        })

        result, _ = await diagnose_pipeline_state(
            query="...", use_case="web_research",
            n_rounds=2, max_rounds=3, stop_reason="stop_done",
            final_extracts=45, positive_extracts=42,
            negative_extracts=3, meta_extracts=0, n_sources=20,
            coverage_distribution={"answered": 4, "partial": 1,
                                   "unanswered": 0, "filter_blocked": 0},
            filter_stats={
                "negative_extract": {
                    "activated": True, "rejected": 3, "total": 45,
                },
            },
            anchor_type="none", anchor_confidence=0.95,
            active_filters=["negative_extract"],
            llm=llm,
        )

        self.assertEqual(result.diagnosis, "successful")
        self.assertTrue(result.is_successful())

    @async_test
    async def test_consistency_check_filter_too_strict_with_low_loss(self):
        """filter_too_strict but loss rate < 50 % → inconsistency, fallback."""
        llm = MockLLM().queue_json({
            "diagnosis": "filter_too_strict",
            "confidence": 0.85,
            "remediation": "Filter X überprüfen",
            "user_message": "Filter zu streng",
            "reasoning": "...",
        })

        # But: the filter loss rate is only 10 % — that is not "too strict"
        result, _ = await diagnose_pipeline_state(
            query="...", use_case="web_research",
            n_rounds=2, max_rounds=3, stop_reason="stop_done",
            final_extracts=18, positive_extracts=18,
            negative_extracts=0, meta_extracts=0, n_sources=20,
            coverage_distribution={"answered": 3, "partial": 1,
                                   "unanswered": 0, "filter_blocked": 0},
            filter_stats={
                "negative_extract": {
                    "activated": True, "rejected": 2, "total": 20,
                },
            },
            anchor_type="none", anchor_confidence=0.9,
            active_filters=["negative_extract"],
            llm=llm,
        )

        # inconsistency recognised → fall back to partial_success
        self.assertEqual(result.diagnosis, "partial_success")
        self.assertTrue(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
