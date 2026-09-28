"""
Tests for `src.pipeline.harvest_parser`.

Acceptance:
  - the extract "Die Quelle bietet keine Datenblätter, sondern nur Marketing"
    is classified as polarity="meta" (a statement about the source), not
    discarded.
  - the extract "Quelle X erwähnt B300-Spezifikationen nicht" is
    classified as polarity="negative".
  - no phrase list: texts that contain typical "no information" phrases
    but carry a [POSITIV] marker are let through as positive.
"""

import unittest

from src.pipeline.harvest_parser import (
    count_by_polarity,
    meta_only,
    parse_harvest_response_with_polarity,
    positive_only,
)


class TestHarvestParserPolarity(unittest.TestCase):

    def test_basic_positive_extract(self):
        response = """[F1] [POSITIV] Fakt: Die DGX B300 verbraucht typisch 14kW
     Kontext: Datasheet, Power-Sektion
     Verlässlichkeit: hoch"""
        extracts = parse_harvest_response_with_polarity(
            response, "https://nvidia.com/x", "DGX Datasheet",
        )
        self.assertEqual(len(extracts), 1)
        self.assertEqual(extracts[0].question_id, "F1")
        self.assertEqual(extracts[0].polarity, "positive")
        self.assertIn("14kW", extracts[0].fact)
        self.assertEqual(extracts[0].reliability, "high")  # normalised code

    def test_negative_extract(self):
        """NEGATIV marker → polarity='negative'."""
        response = """[F2] [NEGATIV] Fakt: Quelle X erwähnt B300-Spezifikationen nicht
     Kontext: Allgemeine NVIDIA-Übersicht
     Verlässlichkeit: hoch"""
        extracts = parse_harvest_response_with_polarity(response, "u", "t")
        self.assertEqual(len(extracts), 1)
        self.assertEqual(extracts[0].polarity, "negative")

    def test_meta_extract(self):
        """META marker for statements about the source.

        'Die Quelle bietet keine Datenblätter, sondern nur Marketing' —
        that is a useful meta finding, not a negative extract.
        """
        response = """[F1] [META] Fakt: Die Quelle bietet keine Datenblätter, sondern nur Marketing
     Kontext: NVIDIA-Marketing-Seite
     Verlässlichkeit: hoch"""
        extracts = parse_harvest_response_with_polarity(response, "u", "t")
        self.assertEqual(len(extracts), 1)
        self.assertEqual(extracts[0].polarity, "meta")

    def test_mixed_polarities_in_one_response(self):
        """Realistic case: three polarities in one source."""
        response = """[F1] [POSITIV] Fakt: TDP 14kW typisch
     Verlässlichkeit: hoch

[F2] [NEGATIV] Fakt: Quelle erwähnt MTBF nicht
     Verlässlichkeit: hoch

[F3] [META] Fakt: Die Quelle ist eine Marketing-Folie, kein Datasheet
     Verlässlichkeit: hoch"""
        extracts = parse_harvest_response_with_polarity(response, "u", "t")
        counts = count_by_polarity(extracts)
        self.assertEqual(counts, {"positive": 1, "negative": 1, "meta": 1})

    def test_missing_marker_defaults_to_positive(self):
        """No marker → POSITIVE (default).

        Important so that harvest responses without markers are not
        discarded as 'unknown' by mistake.
        """
        response = """[F1] Fakt: Etwas ohne Marker
     Verlässlichkeit: hoch"""
        extracts = parse_harvest_response_with_polarity(response, "u", "t")
        self.assertEqual(len(extracts), 1)
        self.assertEqual(extracts[0].polarity, "positive")

    def test_blacklist_phrase_with_positive_marker_kept(self):
        """A POSITIVE extract may contain phrases like "keine".

        A phrase list would discard this extract because it contains the
        word 'keine'. With the marker approach it stays — it IS a positive
        statement about the source.
        """
        response = """[F1] [POSITIV] Fakt: Die offizielle Spec-Tabelle zeigt: keine Unterstützung für PCIe Gen6, dafür Gen5 mit 128 Lanes
     Verlässlichkeit: hoch"""
        extracts = parse_harvest_response_with_polarity(response, "u", "t")
        self.assertEqual(len(extracts), 1)
        self.assertEqual(extracts[0].polarity, "positive")
        # content kept completely
        self.assertIn("keine Unterstützung", extracts[0].fact)

    def test_multiline_fact(self):
        """A fact spans several lines → it is concatenated."""
        response = """[F1] [POSITIV] Fakt: Die DGX B300 verbraucht 14kW typisch
     und 16kW bei voller Last — laut offiziellem Datasheet von NVIDIA.
     Kontext: Power-Sektion
     Verlässlichkeit: hoch"""
        extracts = parse_harvest_response_with_polarity(response, "u", "t")
        self.assertEqual(len(extracts), 1)
        self.assertIn("14kW", extracts[0].fact)
        self.assertIn("16kW", extracts[0].fact)

    def test_count_by_polarity(self):
        from src.pipeline.models import SourceExtract
        extracts = [
            SourceExtract("u", "t", "F1", "x", polarity="positive"),
            SourceExtract("u", "t", "F1", "x", polarity="positive"),
            SourceExtract("u", "t", "F2", "x", polarity="negative"),
            SourceExtract("u", "t", "F3", "x", polarity="meta"),
        ]
        self.assertEqual(
            count_by_polarity(extracts),
            {"positive": 2, "negative": 1, "meta": 1},
        )

    def test_positive_only_filter(self):
        from src.pipeline.models import SourceExtract
        extracts = [
            SourceExtract("u", "t", "F1", "a", polarity="positive"),
            SourceExtract("u", "t", "F1", "b", polarity="negative"),
            SourceExtract("u", "t", "F1", "c", polarity="meta"),
        ]
        pos = positive_only(extracts)
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0].fact, "a")

    def test_meta_only_filter(self):
        from src.pipeline.models import SourceExtract
        extracts = [
            SourceExtract("u", "t", "F1", "a", polarity="positive"),
            SourceExtract("u", "t", "F1", "b", polarity="meta"),
        ]
        m = meta_only(extracts)
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0].fact, "b")

    def test_invalid_marker_treated_as_positive(self):
        """Marker value not in {POSITIV,NEGATIV,META} → treated like no marker."""
        response = """[F1] [BLA] Fakt: Etwas
     Verlässlichkeit: hoch"""
        extracts = parse_harvest_response_with_polarity(response, "u", "t")
        # Marker not recognised → the ID line is parsed as an ordinary fact,
        # polarity stays at the default
        self.assertEqual(len(extracts), 1)
        self.assertEqual(extracts[0].polarity, "positive")


if __name__ == "__main__":
    unittest.main()


def test_english_markers_and_labels_are_parsed():
    """The prompt asks for English markers and labels; German ones stay accepted."""
    from src.pipeline.harvest_parser import parse_harvest_response_with_polarity as parse_harvest_response
    response = """[F1] [POSITIVE] Fact: The DGX B300 draws 14 kW
     Context: datasheet
     Reliability: high

[F2] [NEGATIVE] Fact: The source does not cover pricing
     Context: -
     Reliability: medium

[F3] [META] Fact: The page is marketing material
     Context: -
     Reliability: low"""
    ex = parse_harvest_response(response, "https://example.org", "Example")
    assert [e.polarity for e in ex] == ["positive", "negative", "meta"]
    assert ex[0].fact.startswith("The DGX B300") and "[POSITIVE]" not in ex[0].fact
    assert ex[0].reliability in ("high", "hoch")
