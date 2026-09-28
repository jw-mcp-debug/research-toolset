"""
Tests for the Gradio helpers in src/ui/analysis_runner.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.analysis_runner import (
    build_explainer_preflight_data,
    validate_explainer_inputs,
)


def test_validate_rejects_short_inputs():
    """Preflight rejects inputs that are too short."""
    print("Test: validate — inputs too short ...", end=" ")
    ok, err = validate_explainer_inputs("short", "also short", "medium")
    assert not ok
    assert "Topic" in err or "Audience" in err
    print("✓")


def test_validate_accepts_valid():
    """Preflight accepts valid inputs."""
    print("Test: validate — valid inputs ...", end=" ")
    ok, err = validate_explainer_inputs(
        "How do quantum computers work?",
        "Physics undergraduates without quantum mechanics",
        "medium",
        "self_study",
    )
    assert ok, f"Unexpected error: {err}"
    print("✓")


def test_validate_invalid_length():
    """Preflight rejects an unknown length."""
    print("Test: validate — unknown length ...", end=" ")
    ok, err = validate_explainer_inputs(
        "Wie funktionieren LLMs?",
        "Bachelor-Informatik-Studierende",
        "extrem-lang",  # not in options
    )
    assert not ok
    print("✓")


def test_build_preflight_normalizes():
    """build_preflight sets defaults and trims."""
    print("Test: build_preflight — Defaults ...", end=" ")
    data = build_explainer_preflight_data(
        "  How do LLMs work?  ",
        "  CS undergraduates  ",
        "short",
    )
    assert data["topic"] == "How do LLMs work?"  # trimmed
    assert data["audience"] == "CS undergraduates"  # trimmed
    assert data["length"] == "short"
    assert data.get("purpose")  # default set
    print("✓")


def main():
    print("=" * 60)
    print("Analysis-Runner Helper Tests")
    print("=" * 60)
    tests = [
        test_validate_rejects_short_inputs,
        test_validate_accepts_valid,
        test_validate_invalid_length,
        test_build_preflight_normalizes,
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
            import traceback; traceback.print_exc()
            failed += 1
    print("=" * 60)
    if failed:
        print(f"❌ {failed}/{len(tests)} tests failed")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed")


if __name__ == "__main__":
    main()
