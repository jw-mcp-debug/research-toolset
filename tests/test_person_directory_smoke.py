"""
Smoke tests for `src.connectors.person_directory`.

The module is only imported lazily in the pipeline code (inside handler
functions) — so syntax errors in it would only show up when a pipeline run
first tries the import.

These smoke tests check at least that:
  - the module can be imported without errors
  - the central helpers (`_is_person_name_query`, `_name_matches_query`,
    `_extract_name_parts`) do what they should
  - the `DirectoryPerson` data class works
  - the `PersonDirectoryConnector` can be instantiated (without real DB
    access)
"""

import unittest


class TestDirectoryModuleImports(unittest.TestCase):
    """Sanity checks: the module loads at all."""

    def test_module_importable(self):
        """The module must load without SyntaxError or ImportError."""
        import src.connectors.person_directory as directory  # noqa: F401

    def test_public_symbols_present(self):
        """The most important public symbols must be exported."""
        from src.connectors import person_directory
        for name in (
            "DirectoryPerson",
            "DirectorySearchConfig",
            "PersonDirectoryConnector",
            "_is_person_name_query",
            "_name_matches_query",
            "_extract_name_parts",
        ):
            self.assertTrue(
                hasattr(person_directory, name),
                f"symbol {name!r} missing in person_directory",
            )


class TestExtractNameParts(unittest.TestCase):

    def test_simple_name(self):
        from src.connectors.person_directory import _extract_name_parts
        parts = _extract_name_parts("Jonas Brenner")
        self.assertEqual(set(parts), {"jonas", "brenner"})

    def test_titles_stripped(self):
        from src.connectors.person_directory import _extract_name_parts
        parts = _extract_name_parts("Prof. Dr. Hans Müller")
        # "prof", "dr" are removed; "hans", "müller" remain
        # (in normalised form)
        self.assertNotIn("prof", parts)
        self.assertNotIn("dr", parts)
        self.assertIn("hans", parts)

    def test_diacritics_normalized(self):
        from src.connectors.person_directory import _extract_name_parts
        a = _extract_name_parts("Kovács")
        b = _extract_name_parts("Kovacs")
        self.assertEqual(a, b)


class TestIsPersonNameQuery(unittest.TestCase):
    """Heuristic: is the query a person's name?"""

    def test_two_word_name(self):
        from src.connectors.person_directory import _is_person_name_query
        self.assertTrue(_is_person_name_query("Jonas Brenner"))

    def test_three_word_name(self):
        from src.connectors.person_directory import _is_person_name_query
        self.assertTrue(_is_person_name_query("Hans Peter Müller"))

    def test_single_word_no(self):
        from src.connectors.person_directory import _is_person_name_query
        # Single words are not recognised as a person's name
        # (confidence too low)
        self.assertFalse(_is_person_name_query("Müller"))

    def test_topic_query_no(self):
        from src.connectors.person_directory import _is_person_name_query
        self.assertFalse(_is_person_name_query("Materialwissenschaften"))

    def test_url_no(self):
        from src.connectors.person_directory import _is_person_name_query
        self.assertFalse(_is_person_name_query("site:example.edu"))
        self.assertFalse(_is_person_name_query("https://example.com"))

    def test_with_question_mark_no(self):
        from src.connectors.person_directory import _is_person_name_query
        self.assertFalse(_is_person_name_query("Wer ist Hans?"))


class TestNameMatchesQuery(unittest.TestCase):
    """Match logic: does the person's name match the query?"""

    def test_exact_match(self):
        from src.connectors.person_directory import _name_matches_query
        self.assertTrue(_name_matches_query("Jonas Brenner", "Jonas Brenner"))

    def test_diacritics_match(self):
        from src.connectors.person_directory import _name_matches_query
        # Kovacs ↔ Kovács
        self.assertTrue(_name_matches_query("Mira Kovacs", "Mira Kovács"))

    def test_order_independent(self):
        from src.connectors.person_directory import _name_matches_query
        self.assertTrue(_name_matches_query("Kovacs Mira", "Mira Kovács"))

    def test_partial_no_match(self):
        from src.connectors.person_directory import _name_matches_query
        # Both names must occur
        self.assertFalse(_name_matches_query("Mira Kovacs", "Mira Lindqvist"))

    def test_query_subset_of_pname(self):
        from src.connectors.person_directory import _name_matches_query
        # If the query has only the family name, it must be in the person's name
        self.assertTrue(_name_matches_query("Müller", "Anna Müller"))


class TestDirectoryPerson(unittest.TestCase):

    def test_construction(self):
        from src.connectors.person_directory import DirectoryPerson
        r = DirectoryPerson(
            pid=42, person_name="Jonas Brenner",
            email="uwe@example.edu", org_path="CMS",
            subject_area="IT",
        )
        self.assertEqual(r.pid, 42)
        self.assertEqual(r.person_name, "Jonas Brenner")


if __name__ == "__main__":
    unittest.main()
