"""
Tests for the hard person-name filter of the person directory.

Checks:
- _normalize_name_for_match: remove diacritics
- _extract_name_parts: filter titles and particles
- _is_person_name_query: recognise person queries
- _name_matches_query: hard match with all name parts
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# mock numpy if not available (person_directory imports it)
if "numpy" not in sys.modules:
    sys.modules["numpy"] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectors.person_directory import (  # noqa: E402
    _normalize_name_for_match,
    _extract_name_parts,
    _is_person_name_query,
    _name_matches_query,
)


def test_normalize_removes_diacritics():
    print("Test: normalize removes diacritics ...", end=" ")
    assert _normalize_name_for_match("Kovács") == "kovacs"
    assert _normalize_name_for_match("Müller") == "muller"
    assert _normalize_name_for_match("François") == "francois"
    assert _normalize_name_for_match("Søren") == "soren"
    assert _normalize_name_for_match("Kovacs, Mira") == "kovacs mira"
    print("✓")


def test_normalize_handles_empty():
    print("Test: normalize empty ...", end=" ")
    assert _normalize_name_for_match("") == ""
    assert _normalize_name_for_match(None) == ""
    print("✓")


def test_normalize_compound_names():
    print("Test: normalize compound names ...", end=" ")
    assert _normalize_name_for_match("Müller-Weber") == "muller weber"
    assert _normalize_name_for_match("de Saint-Exupéry") == "de saint exupery"
    print("✓")


def test_extract_name_parts_removes_titles():
    print("Test: extract removes titles ...", end=" ")
    assert _extract_name_parts("Dr. Mira Kovacs") == ["mira", "kovacs"]
    assert _extract_name_parts("Prof. Dr. Anna Müller") == ["anna", "muller"]
    assert _extract_name_parts("Herr Lindqvist") == ["lindqvist"]
    print("✓")


def test_extract_name_parts_removes_particles():
    print("Test: extract removes particles ...", end=" ")
    assert _extract_name_parts("Ludwig von Beethoven") == [
        "ludwig", "beethoven"
    ]
    assert _extract_name_parts("van der Berg") == ["berg"]
    print("✓")


def test_extract_name_parts_short_tokens():
    print("Test: extract filters tokens that are too short ...", end=" ")
    # "X Kovacs" — a one-letter token is removed
    assert _extract_name_parts("X Kovacs") == ["kovacs"]
    print("✓")


def test_is_person_name_query_positive():
    print("Test: is_person recognises names ...", end=" ")
    assert _is_person_name_query("Mira Kovacs")
    assert _is_person_name_query("Kovacs Mira")
    assert _is_person_name_query("Dr. Anna Müller")
    assert _is_person_name_query("Prof. Dr. Anna Lindqvist")
    # A single name is not unambiguously a person's name — it could also
    # be a topical term. Hence the heuristic requires 2-3 parts.
    assert not _is_person_name_query("Prof. Weber")
    assert not _is_person_name_query("Kovacs")  # too ambiguous
    print("✓")


def test_is_person_name_query_negative():
    print("Test: is_person rejects non-names ...", end=" ")
    assert not _is_person_name_query("site:example.edu")
    assert not _is_person_name_query("Was ist KI?")
    assert not _is_person_name_query("https://example.com")
    assert not _is_person_name_query("")
    assert not _is_person_name_query("ein zwei drei vier fünf")  # zu lang
    print("✓")


def test_name_matches_exact():
    print("Test: name_match exact ...", end=" ")
    assert _name_matches_query("Mira Kovacs", "Mira Kovacs")
    assert _name_matches_query("mira kovacs", "Mira Kovacs")
    print("✓")


def test_name_matches_with_diacritics():
    print("Test: name_match with diacritics ...", end=" ")
    # The core case: the user searches "Kovacs", the DB has "Kovács"
    assert _name_matches_query("Mira Kovacs", "Mira Kovács")
    assert _name_matches_query("Kovacs Mira", "Mira Kovács")
    assert _name_matches_query("Müller", "Anna Mueller") is False  # ae != ü
    # but: "Müller" in the query matches "Anna Müller" in the DB
    assert _name_matches_query("Müller", "Anna Müller")
    print("✓")


def test_name_matches_reversed_order():
    print("Test: name_match order irrelevant ...", end=" ")
    assert _name_matches_query("Mira Kovacs", "Kovacs, Mira")
    assert _name_matches_query("Kovacs Mira", "Mira Kovács")
    print("✓")


def test_name_matches_rejects_fuzzy():
    """The core case: without the filter, a fuzzy match on the given name or
    the family name alone would be enough, returning many wrong persons.
    Now BOTH parts must occur.
    """
    print("Test: name_match rejects fuzzy matches ...", end=" ")
    # Wrong hits a fuzzy match would return
    assert not _name_matches_query("Mira Kovacs", "Mira Lindqvist")
    assert not _name_matches_query("Mira Kovacs", "Miriam Roth")
    assert not _name_matches_query("Mira Kovacs", "Dr. Clara Winter")
    assert not _name_matches_query("Mira Kovacs", "Mira Vogel")
    assert not _name_matches_query("Mira Kovacs", "Dr. Mira Castell")
    # But the one correct hit matches
    assert _name_matches_query("Mira Kovacs", "Mira Kovács")
    print("✓")


def test_name_matches_single_part():
    """A single name part in the query matches if it occurs in person_name."""
    print("Test: name_match single part ...", end=" ")
    assert _name_matches_query("Kovacs", "Mira Kovács")
    assert not _name_matches_query("Lindqvist", "Mira Kovacs")
    print("✓")


def main():
    print("=" * 60)
    print("Tests: person-name filter of the person directory")
    print("=" * 60)

    tests = [
        test_normalize_removes_diacritics,
        test_normalize_handles_empty,
        test_normalize_compound_names,
        test_extract_name_parts_removes_titles,
        test_extract_name_parts_removes_particles,
        test_extract_name_parts_short_tokens,
        test_is_person_name_query_positive,
        test_is_person_name_query_negative,
        test_name_matches_exact,
        test_name_matches_with_diacritics,
        test_name_matches_reversed_order,
        test_name_matches_rejects_fuzzy,
        test_name_matches_single_part,
    ]

    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    if failed:
        print(f"❌ {failed}/{len(tests)} tests failed")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
