"""Tests for PreflightChecker.normalize() and the autofill from the chat history."""

import asyncio

import tests.conftest  # noqa: F401  (dependency stubs)

from src.pipeline.analysis_pipeline import USE_CASE_REGISTRY, Requirement
from src.ui.preflight_autofill import (
    build_autofill_prompt,
    format_conversation,
    format_requirements,
    sanitize_suggestions,
    suggest_preflight_values,
)


# ── normalize() after validate() ────────────────────────────────────────

def test_every_checker_has_normalize():
    """gradio_app calls normalize() directly after validate()."""
    for use_case, entry in USE_CASE_REGISTRY.items():
        checker = entry["preflight"]
        assert hasattr(checker, "normalize"), (
            f"{use_case}: PreflightChecker without normalize() — "
            f"exactly this caused 'object has no attribute normalize'"
        )


def test_normalize_returns_independent_copy():
    checker = USE_CASE_REGISTRY["decision_analysis"]["preflight"]
    inputs = {"decision": "Welches CRM?", "options": "A\nB",
              "criteria": "Kosten"}
    out = checker.normalize(inputs)
    assert out == inputs
    out["decision"] = "geaendert"
    assert inputs["decision"] == "Welches CRM?", "normalize mutates the input"


def test_validate_then_normalize_succeeds():
    """The production flow: validate() passes, then normalize()."""
    checker = USE_CASE_REGISTRY["decision_analysis"]["preflight"]
    inputs = {
        "decision": "Welches CRM-System sollen wir einfuehren?",
        "options": "Salesforce: teuer\nHubSpot: guenstiger",
        "criteria": "Kosten\nIntegration",
    }
    ok, errors = checker.validate(inputs)
    assert ok, errors
    assert checker.normalize(inputs)["decision"].startswith("Welches CRM")


# ── Autofill: building the prompt ───────────────────────────────────────

REQS = [
    Requirement(field="decision", label="Entscheidungsfrage", kind="text"),
    Requirement(field="options", label="Optionen", kind="textarea"),
    Requirement(field="horizon", label="Horizont", kind="choice",
                choices=["kurz", "mittel", "lang"]),
    Requirement(field="note", label="Notiz", kind="text",
                required=False, max_length=20),
]


def test_prompt_lists_fields_and_choices():
    text = format_requirements(REQS)
    assert '"decision"' in text and '"horizon"' in text
    assert "kurz | mittel | lang" in text
    assert "optional" in text          # note is required=False
    assert "At most 20 characters" in text


def test_prompt_forbids_invention():
    prompt = build_autofill_prompt(REQS, "Person: Hallo")
    assert "Invent nothing" in prompt
    assert "empty string" in prompt


def test_conversation_uses_roles_and_appends_draft():
    history = [
        {"role": "user", "content": "Wir suchen ein CRM."},
        {"role": "assistant", "content": "Welche Optionen?"},
        {"role": "user", "content": ""},          # empty -> ignore
        "kaputt",                                   # not a dict -> ignore
    ]
    text = format_conversation(history, extra_text="Budget 50k")
    assert "Person: Wir suchen ein CRM." in text
    assert "Assistant: Welche Optionen?" in text
    assert text.rstrip().endswith("Person: Budget 50k")


def test_conversation_truncates_from_the_front():
    history = [{"role": "user", "content": "A" * 40_000},
               {"role": "user", "content": "LETZTE NACHRICHT"}]
    text = format_conversation(history)
    assert "LETZTE NACHRICHT" in text, "the end of the history must be kept"
    assert "shortened" in text


# ── Autofill: cleaning up the LLM answer ──────────────────────────

def test_sanitize_keeps_only_known_fields():
    out = sanitize_suggestions(
        {"decision": "Welches CRM?", "erfunden": "Wert"}, REQS)
    assert out == {"decision": "Welches CRM?"}


def test_sanitize_drops_empty_values():
    out = sanitize_suggestions(
        {"decision": "  ", "options": "", "horizon": None}, REQS)
    assert out == {}, "empty fields must not overwrite anything"


def test_sanitize_enforces_choices():
    assert sanitize_suggestions({"horizon": "sehr lang"}, REQS) == {}
    assert sanitize_suggestions({"horizon": "mittel"}, REQS)["horizon"] == "mittel"
    # Capitalisation is tolerated, but normalised to the allowed value
    assert sanitize_suggestions({"horizon": "Mittel"}, REQS)["horizon"] == "mittel"


def test_sanitize_joins_lists_to_lines():
    out = sanitize_suggestions({"options": ["Salesforce", "HubSpot"]}, REQS)
    assert out["options"] == "Salesforce\nHubSpot"


def test_sanitize_truncates_to_max_length():
    out = sanitize_suggestions({"note": "x" * 100}, REQS)
    assert len(out["note"]) == 20


def test_sanitize_survives_garbage():
    assert sanitize_suggestions("kein dict", REQS) == {}
    assert sanitize_suggestions(None, REQS) == {}


# ── Autofill: end to end against a fake LLM ───────────────────────

class FakeChecker:
    def get_requirements(self):
        return REQS


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    async def harvest_complete(self, messages, max_tokens=None):
        self.prompts.append(messages[0]["content"])
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


HISTORY = [{"role": "user", "content": (
    "Wir muessen uns zwischen Salesforce und HubSpot entscheiden. "
    "Wichtig sind uns vor allem die Kosten und die Integration."
)}]


def test_suggest_happy_path():
    llm = FakeLLM('```json\n{"decision": "Welches CRM?", '
                  '"options": "Salesforce\\nHubSpot", "horizon": "mittel"}\n```')
    values, msg = asyncio.run(suggest_preflight_values(
        FakeChecker(), HISTORY, llm))
    assert msg == ""
    assert values["decision"] == "Welches CRM?"
    assert values["horizon"] == "mittel"
    assert "note" not in values


def test_suggest_refuses_on_thin_conversation():
    llm = FakeLLM('{"decision": "geraten"}')
    values, msg = asyncio.run(suggest_preflight_values(
        FakeChecker(), [{"role": "user", "content": "hi"}], llm))
    assert values == {} and "Gesprächsverlauf" in msg
    assert llm.prompts == [], "do not ask at all with too little history"


def test_suggest_handles_unparseable_answer():
    values, msg = asyncio.run(suggest_preflight_values(
        FakeChecker(), HISTORY, FakeLLM("Klar, gerne! Also...")))
    assert values == {} and msg


def test_suggest_handles_llm_error():
    values, msg = asyncio.run(suggest_preflight_values(
        FakeChecker(), HISTORY, FakeLLM(RuntimeError("boom"))))
    assert values == {} and "RuntimeError" in msg


def test_suggest_reports_when_nothing_derivable():
    values, msg = asyncio.run(suggest_preflight_values(
        FakeChecker(), HISTORY, FakeLLM('{"decision": "", "options": ""}')))
    assert values == {} and "ableiten" in msg


# ── Autofill on clicking "Start research" ───────────────────

from src.ui.preflight_autofill import (  # noqa: E402
    empty_required_fields, plan_form_autofill,
)


def test_empty_required_fields_detects_blanks():
    # "kurz" fills horizon -> only decision and options are empty
    assert empty_required_fields(REQS, ["", "  ", "kurz", ""]) == [
        "decision", "options"]


def test_none_counts_as_empty():
    """str(None) would be 'None' — the classic mistake at this point."""
    assert "decision" in empty_required_fields(REQS, [None, "x", "kurz", ""])


def test_missing_positions_count_as_empty():
    assert empty_required_fields(REQS, []) == [
        "decision", "options", "horizon"]


def test_optional_fields_are_not_required():
    """'note' is required=False and must not block the run."""
    assert "note" not in empty_required_fields(REQS, ["", "", "", ""])


def test_full_form_needs_no_autofill():
    assert empty_required_fields(REQS, ["D", "A\nB", "kurz", ""]) == []
    assert plan_form_autofill(
        REQS, ["D", "A\nB", "kurz", ""], {"decision": "Vorschlag"}) == {}


def test_autofill_never_overwrites_user_input():
    """A suggestion must never replace a deliberate input."""
    plan = plan_form_autofill(
        REQS,
        ["Meine eigene Frage", "", "", ""],
        {"decision": "LLM-Vorschlag", "options": "A\nB"},
    )
    assert 0 not in plan, "field 0 was filled in"
    assert plan == {1: "A\nB"}


def test_autofill_maps_to_correct_indices():
    plan = plan_form_autofill(
        REQS, ["", "", "", ""],
        {"decision": "D", "horizon": "mittel"})
    assert plan == {0: "D", 2: "mittel"}


def test_autofill_ignores_suggestions_for_filled_fields():
    plan = plan_form_autofill(
        REQS, ["", "vorhanden", "", ""], {"options": "X", "decision": "D"})
    assert plan == {0: "D"}
