"""
Tests for RunningContext.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.running_context import RunningContext


def test_basic_add_and_get():
    """Add and retrieve."""
    print("Test: Basic add/get ...", end=" ")
    ctx = RunningContext(max_tokens=4000)
    ctx.add("intro", "Einführung in das Thema.")
    ctx.add("part1", "Erster Hauptteil.")
    result = ctx.get_context()
    assert "Einführung" in result
    assert "Hauptteil" in result
    assert ctx.section_count == 2
    print("✓")


def test_empty_context():
    """Empty context."""
    print("Test: Empty context ...", end=" ")
    ctx = RunningContext()
    assert ctx.get_context() == ""
    assert ctx.section_count == 0
    print("✓")


def test_replace_section():
    """Replace a section with the same ID."""
    print("Test: Replace section ...", end=" ")
    ctx = RunningContext()
    ctx.add("s1", "alter Inhalt")
    ctx.add("s1", "neuer Inhalt")
    assert ctx.section_count == 1
    assert "neuer" in ctx.get_context()
    assert "alter" not in ctx.get_context()
    print("✓")


def test_up_to_section():
    """Filter up to a section."""
    print("Test: up_to_section ...", end=" ")
    ctx = RunningContext()
    ctx.add("a", "Alpha")
    ctx.add("b", "Beta")
    ctx.add("c", "Gamma")
    result = ctx.get_context(up_to_section="c")
    assert "Alpha" in result
    assert "Beta" in result
    assert "Gamma" not in result
    print("✓")


def test_exclude_sections():
    """Exclude sections."""
    print("Test: exclude ...", end=" ")
    ctx = RunningContext()
    ctx.add("a", "Alpha")
    ctx.add("b", "Beta")
    ctx.add("c", "Gamma")
    result = ctx.get_context(exclude=["b"])
    assert "Alpha" in result
    assert "Beta" not in result
    assert "Gamma" in result
    print("✓")


def test_truncation_under_budget():
    """Truncation kicks in at the token limit."""
    print("Test: Token-Truncation ...", end=" ")
    ctx = RunningContext(max_tokens=20)  # very small, ~80 chars budget
    for i in range(10):
        ctx.add(f"s{i}", f"Das ist ein längerer Text für Section {i}.")
    result = ctx.get_context()
    # Should be clearly shorter than untruncated
    full_length = sum(len(f"Das ist ein längerer Text für Section {i}.") for i in range(10))
    assert len(result) < full_length, f"truncation does not work: {len(result)} chars (of {full_length})"
    # The most recent sections must be included (content, not just the header)
    assert "s9" in result, f"most recent section missing: {result[:200]}"
    print(f"✓ ({len(result)} instead of {full_length} chars)")


def test_serialize_deserialize():
    """Persistence."""
    print("Test: Serialize/Deserialize ...", end=" ")
    ctx = RunningContext(max_tokens=2000)
    ctx.add("a", "Inhalt A", full_text="Voller Text A")
    ctx.add("b", "Inhalt B")

    data = ctx.serialize()
    restored = RunningContext.deserialize(data)

    assert restored.section_count == 2
    assert restored.section_ids == ["a", "b"]
    assert restored.max_tokens == 2000
    print("✓")


def test_clear():
    """clear() removes all sections."""
    print("Test: clear() ...", end=" ")
    ctx = RunningContext()
    ctx.add("a", "x")
    ctx.add("b", "y")
    ctx.clear()
    assert ctx.section_count == 0
    print("✓")


def main():
    print("=" * 60)
    print("RunningContext Tests")
    print("=" * 60)
    tests = [
        test_basic_add_and_get,
        test_empty_context,
        test_replace_section,
        test_up_to_section,
        test_exclude_sections,
        test_truncation_under_budget,
        test_serialize_deserialize,
        test_clear,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print("=" * 60)
    if failed:
        print(f"❌ {failed}/{len(tests)} tests failed")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
