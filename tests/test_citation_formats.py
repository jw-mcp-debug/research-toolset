"""
Tests for the citation formats (APA 7, DIN 1505-2).

Regression tests:
  - dict authors ({"family", "given"}) correctly become "Müller, A."
  - dict authors ({"name"}) likewise
  - dict authors ({"display_name"}, OpenAlex) likewise
  - a string author is not split into characters
  - et al. logic after 20 authors
  - empty fields are omitted without double full stops/commas
"""

import unittest

from src.pipeline.literature_check import (
    _build_apa_string,
    _build_din_string,
    _format_apa_authors,
)


class TestApaAuthors(unittest.TestCase):

    def test_dict_with_family_and_given(self):
        out = _format_apa_authors([
            {"family": "Müller", "given": "Anna"},
        ])
        self.assertEqual(out, "Müller, A.")

    def test_dict_with_name_field(self):
        out = _format_apa_authors([{"name": "Müller, Anna"}])
        self.assertIn("Müller", out)

    def test_dict_with_display_name(self):
        """OpenAlex-Format."""
        out = _format_apa_authors([{"display_name": "Hans Müller"}])
        self.assertEqual(out, "Müller, H.")

    def test_two_authors_uses_ampersand(self):
        out = _format_apa_authors([
            {"family": "Müller", "given": "Anna"},
            {"family": "Schmidt", "given": "Bernd"},
        ])
        self.assertEqual(out, "Müller, A. & Schmidt, B.")

    def test_three_to_twenty_authors(self):
        out = _format_apa_authors([
            {"family": f"A{i}", "given": "X"} for i in range(5)
        ])
        # APA 7: up to 20 name all, with ", &" before the last
        self.assertIn(", & A4, X.", out)

    def test_more_than_twenty_authors_uses_ellipsis(self):
        """APA 7 rule: after 19 a '... LastName' before the last."""
        out = _format_apa_authors([
            {"family": f"A{i:02d}", "given": "X"} for i in range(25)
        ])
        # first 19, then "... A24, X."
        self.assertIn("... A24, X.", out)
        # A20 not included
        self.assertNotIn("A20", out)

    def test_string_author_not_treated_as_iterable(self):
        """Regression: 'Müller, A.' must not be split into characters."""
        out = _format_apa_authors("Müller, A.")
        self.assertEqual(out, "Müller, A.")

    def test_empty_list(self):
        self.assertEqual(_format_apa_authors([]), "")

    def test_none(self):
        self.assertEqual(_format_apa_authors(None), "")

    def test_missing_given_only_family(self):
        out = _format_apa_authors([{"family": "Müller"}])
        self.assertEqual(out, "Müller")


class TestApaFullString(unittest.TestCase):

    def test_journal_article_full(self):
        out = _build_apa_string(
            authors=[{"family": "Müller", "given": "Anna"}],
            year="2023", title="Studie",
            journal="Journal X", volume="10", issue="2",
            pages="1-15", doi="10.1234/abc",
        )
        # Expected: "Müller, A. (2023) Studie. *Journal X*, *10*(2), 1-15. https://doi.org/..."
        self.assertIn("Müller, A.", out)
        self.assertIn("(2023)", out)
        self.assertIn("Studie", out)
        self.assertIn("*Journal X*", out)
        self.assertIn("*10*(2)", out)
        self.assertIn("1-15", out)
        self.assertIn("https://doi.org/10.1234/abc", out)

    def test_no_author_starts_with_year(self):
        out = _build_apa_string(
            authors=[], year="2020", title="Anon",
            journal="", volume="", issue="", pages="", doi="",
        )
        self.assertTrue(out.startswith("("))
        self.assertIn("2020", out)
        self.assertIn("Anon", out)

    def test_doi_link_format(self):
        out = _build_apa_string(
            authors=[{"family": "X", "given": "Y"}],
            year="2020", title="t",
            journal="", volume="", issue="", pages="",
            doi="10.1000/test",
        )
        self.assertIn("https://doi.org/10.1000/test", out)


class TestDin(unittest.TestCase):

    def test_din_basic(self):
        out = _build_din_string(
            authors=[{"family": "Müller", "given": "Anna"}],
            year="2023", title="Buch",
            journal="", volume="", issue="", pages="",
            doi="", publisher="Springer", isbn="",
            entry_type="book",
        )
        self.assertIn("Müller", out)
        self.assertIn("Buch", out)
        self.assertIn("2023", out)


if __name__ == "__main__":
    unittest.main()
