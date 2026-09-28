"""
Tests for `src.pipeline.query_utils`.

  - `coerce_search_terms` accepts the four LLM output forms seen in
    practice (flat list, dict per language, list of dicts, single string).
  - `detect_term_language` assigns umlaut terms to DE, others to EN.
  - `build_terms_by_lang` aggregates over several questions and prefers
    `search_terms_by_lang` over `search_terms`.
  - robust against edge cases: empty lists, None values, invalid types.
"""

import unittest
from dataclasses import dataclass, field

from src.pipeline.query_utils import (
    build_terms_by_lang,
    coerce_search_terms,
    detect_term_language,
)


# Minimal ResearchQuestion stub for the tests — so that we do not have
# to instantiate the real model with all its fields
@dataclass
class _Q:
    search_terms: list = field(default_factory=list)
    search_terms_by_lang: dict = field(default_factory=dict)
    search_langs: list = field(default_factory=lambda: ["de", "en"])


class TestDetectTermLanguage(unittest.TestCase):

    def test_umlaut_is_german(self):
        self.assertEqual(detect_term_language("Stromaufnahme"), "en")  # no umlaut
        self.assertEqual(detect_term_language("Verlässlichkeit"), "de")
        self.assertEqual(detect_term_language("Größe"), "de")
        self.assertEqual(detect_term_language("Übertragung"), "de")

    def test_function_word_is_german(self):
        self.assertEqual(detect_term_language("der schnellste Server"), "de")
        self.assertEqual(detect_term_language("für Forschung"), "de")

    def test_pure_english_is_english(self):
        self.assertEqual(detect_term_language("machine learning"), "en")
        self.assertEqual(detect_term_language("DGX B300 power"), "en")

    def test_neutral_term_defaults_to_english(self):
        # product names without language-specific markers → en default
        self.assertEqual(detect_term_language("DGX B300"), "en")
        self.assertEqual(detect_term_language("BIFOLD"), "en")
        self.assertEqual(detect_term_language("Jonas Brenner"), "en")

    def test_empty_or_invalid(self):
        self.assertEqual(detect_term_language(""), "en")
        self.assertEqual(detect_term_language("   "), "en")
        self.assertEqual(detect_term_language(None), "en")  # type: ignore


class TestCoerceSearchTerms(unittest.TestCase):

    def test_form_a_flat_list(self):
        flat, by_lang = coerce_search_terms([
            "DGX B300 power", "Stromaufnahme Server",
        ])
        self.assertEqual(flat, ["DGX B300 power", "Stromaufnahme Server"])
        # languages via the heuristic; "Stromaufnahme" without an umlaut → en default
        # (not ideal, but consistent — the question level corrects it)
        self.assertIn("en", by_lang)

    def test_form_b_list_of_dicts(self):
        raw = [
            {"term": "DGX B300 Stromaufnahme", "lang": "de"},
            {"term": "DGX B300 power consumption", "lang": "en"},
        ]
        flat, by_lang = coerce_search_terms(raw)
        self.assertEqual(len(flat), 2)
        self.assertEqual(by_lang["de"], ["DGX B300 Stromaufnahme"])
        self.assertEqual(by_lang["en"], ["DGX B300 power consumption"])

    def test_form_c_dict_per_language(self):
        raw = {
            "de": ["term1", "term2"],
            "en": ["term3"],
        }
        flat, by_lang = coerce_search_terms(raw)
        self.assertEqual(len(flat), 3)
        self.assertEqual(by_lang["de"], ["term1", "term2"])
        self.assertEqual(by_lang["en"], ["term3"])

    def test_form_d_single_string(self):
        flat, by_lang = coerce_search_terms("DGX B300")
        self.assertEqual(flat, ["DGX B300"])

    def test_form_e_empty(self):
        for empty in (None, [], {}, ""):
            flat, by_lang = coerce_search_terms(empty)
            self.assertEqual(flat, [])
            self.assertEqual(by_lang, {})

    def test_dedup_within_call(self):
        flat, _ = coerce_search_terms(["term1", "term1", "term2", "term1"])
        self.assertEqual(flat, ["term1", "term2"])

    def test_language_normalization(self):
        """LLMs sometimes answer with 'german'/'deutsch' instead of 'de' — we normalise."""
        raw = [
            {"term": "x", "lang": "german"},
            {"term": "y", "lang": "deu"},
            {"term": "z", "lang": "english"},
        ]
        _, by_lang = coerce_search_terms(raw)
        self.assertIn("de", by_lang)
        self.assertIn("en", by_lang)
        self.assertEqual(set(by_lang["de"]), {"x", "y"})
        self.assertEqual(by_lang["en"], ["z"])

    def test_invalid_items_skipped(self):
        raw = [
            "valid",
            123,                          # int → ignored
            None,                         # None → ignored
            {"no_term_field": "x"},       # dict without a term → with the heuristic
            {"term": "ok", "lang": "de"},
        ]
        flat, by_lang = coerce_search_terms(raw)
        self.assertIn("valid", flat)
        self.assertIn("ok", flat)
        self.assertIn("x", flat)  # the only string value is taken

    def test_nested_dict_with_term_key(self):
        """Some LLMs nest: {term: {value: '...'}}"""
        raw = [{"term": "echter Term", "lang": "de"}]
        flat, by_lang = coerce_search_terms(raw)
        self.assertEqual(flat, ["echter Term"])
        self.assertEqual(by_lang["de"], ["echter Term"])


class TestBuildTermsByLang(unittest.TestCase):

    def test_uses_structured_when_available(self):
        """search_terms_by_lang is set → it is preferred."""
        q = _Q(
            search_terms_by_lang={
                "de": ["term DE"],
                "en": ["term EN"],
            },
            search_terms=["fallback"],   # should be ignored
        )
        terms_by_lang, all_langs = build_terms_by_lang([q])

        self.assertEqual(terms_by_lang["de"], ["term DE"])
        self.assertEqual(terms_by_lang["en"], ["term EN"])
        self.assertNotIn("fallback", terms_by_lang.get("de", []))
        self.assertNotIn("fallback", terms_by_lang.get("en", []))

    def test_falls_back_to_search_terms(self):
        """Without search_terms_by_lang: split heuristically."""
        q = _Q(
            search_terms=["Größe", "machine learning"],
            search_langs=["de", "en"],
        )
        terms_by_lang, all_langs = build_terms_by_lang([q])

        # Größe → de (umlaut), machine learning → en
        self.assertIn("Größe", terms_by_lang["de"])
        self.assertIn("machine learning", terms_by_lang["en"])

    def test_aggregates_over_multiple_questions(self):
        q1 = _Q(search_terms_by_lang={"de": ["a"], "en": ["b"]})
        q2 = _Q(search_terms_by_lang={"de": ["c"], "en": ["d"]})

        terms_by_lang, all_langs = build_terms_by_lang([q1, q2])

        self.assertEqual(set(terms_by_lang["de"]), {"a", "c"})
        self.assertEqual(set(terms_by_lang["en"]), {"b", "d"})
        self.assertEqual(all_langs, {"de", "en"})

    def test_dedup_across_questions(self):
        q1 = _Q(search_terms_by_lang={"de": ["term"]})
        q2 = _Q(search_terms_by_lang={"de": ["term"]})

        terms_by_lang, _ = build_terms_by_lang([q1, q2])

        self.assertEqual(terms_by_lang["de"], ["term"])  # only in it once

    def test_search_langs_appears_even_without_terms(self):
        """The declared languages are reported."""
        q = _Q(
            search_terms=[],
            search_terms_by_lang={},
            search_langs=["de", "en", "fr"],
        )
        _, all_langs = build_terms_by_lang([q])
        # all three declared languages, even if empty
        self.assertEqual(all_langs, {"de", "en", "fr"})

    def test_heuristic_respects_search_langs(self):
        """If the question only declares DE, an English-looking term belongs in DE too."""
        q = _Q(
            search_terms=["machine learning"],   # heuristically: en
            search_langs=["de"],                  # only DE declared
        )
        terms_by_lang, _ = build_terms_by_lang([q])

        # the term ends up in DE (default_lang), not in EN
        self.assertIn("machine learning", terms_by_lang["de"])
        self.assertNotIn("en", terms_by_lang)

    def test_empty_questions(self):
        terms_by_lang, all_langs = build_terms_by_lang([])
        self.assertEqual(terms_by_lang, {})
        self.assertEqual(all_langs, set())


if __name__ == "__main__":
    unittest.main()
