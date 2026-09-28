"""
Tests for BibTeX generation against known edge cases:
  - `%` in the title would turn the rest of the line into a comment
  - full stops in BibTeX keys (e.g. `n.d.`) are not allowed
  - non-ASCII characters in keys (`García`) break many tools
  - unbalanced braces in the title break the parser
  - `&` is a LaTeX special character
"""

import unittest

from src.pipeline.literature_check import (
    _bibtex_escape,
    _bibtex_key_sanitize,
    format_bibtex_entry,
)


class _StubEntry:
    """Minimal LiteratureEntry stub."""
    def __init__(self, **kw):
        for k in ("authors", "year", "title", "journal", "volume",
                  "issue", "pages", "doi", "publisher", "isbn",
                  "url", "entry_type", "id"):
            default = [] if k == "authors" else ("" if k != "id" else 0)
            setattr(self, k, kw.get(k, default))
        if not self.entry_type:
            self.entry_type = "article"


class TestBibtexEscape(unittest.TestCase):

    def test_percent_escaped(self):
        """% is escaped, otherwise the rest of the line is a comment."""
        self.assertEqual(_bibtex_escape("50% Wachstum"), r"50\% Wachstum")

    def test_ampersand_escaped(self):
        self.assertEqual(_bibtex_escape("A & B"), r"A \& B")

    def test_dollar_escaped(self):
        self.assertEqual(_bibtex_escape("Cost $10"), r"Cost \$10")

    def test_underscore_escaped(self):
        self.assertEqual(_bibtex_escape("var_name"), r"var\_name")

    def test_hash_escaped(self):
        self.assertEqual(_bibtex_escape("a#b"), r"a\#b")

    def test_balanced_braces_kept(self):
        self.assertEqual(
            _bibtex_escape("Title with {protected} part"),
            "Title with {protected} part",
        )

    def test_unbalanced_braces_replaced(self):
        """Unbalanced { or } become (...)."""
        self.assertEqual(_bibtex_escape("Set {x : x > 0"), "Set (x : x > 0")
        self.assertEqual(_bibtex_escape("Some} text"), "Some) text")

    def test_empty_string(self):
        self.assertEqual(_bibtex_escape(""), "")

    def test_none_safe(self):
        self.assertEqual(_bibtex_escape(None), "")


class TestBibtexKeySanitize(unittest.TestCase):

    def test_ascii_letters(self):
        self.assertEqual(_bibtex_key_sanitize("Smith"), "smith")

    def test_german_umlauts(self):
        self.assertEqual(_bibtex_key_sanitize("Müller"), "muller")
        self.assertEqual(_bibtex_key_sanitize("Größe"), "grosse")

    def test_spanish_accents(self):
        self.assertEqual(_bibtex_key_sanitize("García"), "garcia")
        self.assertEqual(_bibtex_key_sanitize("López"), "lopez")

    def test_dots_removed(self):
        """Full stops are not allowed in BibTeX keys."""
        self.assertEqual(_bibtex_key_sanitize("n.d."), "nd")
        self.assertEqual(_bibtex_key_sanitize("U.S.A."), "usa")

    def test_hyphens_kept(self):
        """Hyphens are allowed in keys."""
        self.assertEqual(_bibtex_key_sanitize("Smith-Jones"), "smith-jones")

    def test_slashes_removed(self):
        self.assertEqual(_bibtex_key_sanitize("Über/Unter"), "uberunter")

    def test_empty_string_fallback(self):
        self.assertEqual(_bibtex_key_sanitize(""), "anon")

    def test_only_special_chars_fallback(self):
        self.assertEqual(_bibtex_key_sanitize("..."), "anon")


class TestFormatBibtexEntry(unittest.TestCase):
    """Integration tests against these edge cases."""

    def test_title_with_percent_does_not_truncate(self):
        """CRITICAL: % in the title must not turn the rest into a comment."""
        e = _StubEntry(authors=["X"], year="2024",
                       title="Studie über 50% Wachstum")
        bib = format_bibtex_entry(e)
        self.assertIn(r"\%", bib)
        # The whole title must be kept
        self.assertIn("Wachstum", bib)

    def test_anonymous_entry_has_valid_key(self):
        """Without authors/year the key must still be BibTeX-conformant."""
        e = _StubEntry(authors=[], year="", title="Anonym")
        bib = format_bibtex_entry(e)
        # key on the first line
        first_line = bib.split("\n", 1)[0]
        # full stops or special characters forbidden
        self.assertNotIn(".", first_line)
        # Format: @article{KEY,
        self.assertTrue(first_line.startswith("@"))

    def test_non_ascii_author_produces_ascii_key(self):
        """The key may only contain ASCII — non-ASCII breaks tools."""
        e = _StubEntry(authors=["García López, María"], year="2023",
                       title="Test")
        bib = format_bibtex_entry(e)
        first_line = bib.split("\n", 1)[0]
        # key between { and ,
        key = first_line.split("{", 1)[1].rstrip(",")
        self.assertTrue(key.isascii(),
                        f"key {key!r} contains non-ASCII")
        self.assertIn("garcia", key)

    def test_url_ampersand_escaped(self):
        e = _StubEntry(authors=["X"], year="2024", title="Webseite",
                       url="https://example.com/page?q=1&x=2",
                       entry_type="web")
        bib = format_bibtex_entry(e)
        self.assertIn(r"\&", bib)

    def test_doi_prefix_stripped(self):
        """The DOI URL prefix is removed."""
        e = _StubEntry(authors=["X"], year="2024", title="T",
                       doi="https://doi.org/10.1234/abc")
        bib = format_bibtex_entry(e)
        self.assertIn("10.1234/abc", bib)
        self.assertNotIn("doi.org/10", bib)

    def test_doi_dx_doi_org_stripped(self):
        e = _StubEntry(authors=["X"], year="2024", title="T",
                       doi="https://dx.doi.org/10.1234/abc")
        bib = format_bibtex_entry(e)
        self.assertIn("10.1234/abc", bib)
        self.assertNotIn("dx.doi.org", bib)

    def test_unbalanced_braces_in_title_safe(self):
        """Unbalanced { in the title would otherwise break the parser."""
        e = _StubEntry(authors=["X"], year="2024",
                       title="Set {x : x > 0")
        bib = format_bibtex_entry(e)
        # corrected braces (1 opening, 1 closing: the outer {} of the field value)
        # in the content itself: no braces
        title_line = [l for l in bib.split("\n") if "title" in l][0]
        # the content between {{ and }} would be balanced, but we expect (...)
        self.assertIn("(x : x > 0", title_line)


if __name__ == "__main__":
    unittest.main()
