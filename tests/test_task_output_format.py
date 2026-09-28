"""Tests for separating the choice of model from the output format.

If `_llm_complete` decided by `model_preference` whether to make a JSON
call, every task with `model_preference="primary"` would get a dict back
although its prompt asks for prose — and `_build_final_report` would write
the dict into the report as a Python repr via `str()`:

    {'empfehlung': '...', 'begruendung': [...], 'risikoabwaegung': '...'}
"""

import asyncio
import json

import tests.conftest  # noqa: F401  (dependency stubs)

from src.pipeline.analysis_pipeline import (
    AnalysisPipelineRunner,
    Decomposer,
    AnalysisLayerNode,
    _as_text,
)
from src.pipeline.models import SubTask, TaskPlan, TaskResult, TaskState


# ── _as_text: the safeguard ─────────────────────────────────────

def test_as_text_passes_strings_through():
    assert _as_text("# Bericht\n\nInhalt") == "# Bericht\n\nInhalt"
    assert _as_text("") == ""
    assert _as_text(None) == ""


def test_as_text_renders_dict_as_json_not_repr():
    out = _as_text({"empfehlung": "Lokale Infrastruktur", "punkte": ["a", "b"]})
    assert "'empfehlung'" not in out, "no Python repr may result"
    assert '"empfehlung"' in out
    assert json.loads(out)["punkte"] == ["a", "b"]


def test_as_text_survives_unserializable():
    assert _as_text(object()) != ""


# ── SubTask: format and model are decoupled ────────────────────

def test_output_format_defaults_to_text():
    t = SubTask(id="X", phase="p", description="d", prompt_template="t")
    assert t.output_format == "text"


def test_model_and_format_are_independent():
    t = SubTask(id="X", phase="p", description="d", prompt_template="t",
                model_preference="harvest", output_format="json")
    assert (t.model_preference, t.output_format) == ("harvest", "json")


# ── _llm_complete: Dispatch ───────────────────────────────────────

class FakeLLM:
    def __init__(self):
        self.calls = []

    async def primary_complete(self, messages, **kw):
        self.calls.append("primary_text")
        return "## Empfehlung\n\nProsa-Bericht."

    async def harvest_complete(self, messages, **kw):
        self.calls.append("harvest_text")
        return "Kurze Bewertung."

    async def primary_complete_json(self, messages, **kw):
        self.calls.append("primary_json")
        return {"empfehlung": "X"}

    async def harvest_complete_json(self, messages, **kw):
        self.calls.append("harvest_json")
        return {"score": 3}


def _runner(llm):
    node = AnalysisLayerNode.__new__(AnalysisLayerNode)
    node.llm_client = llm
    return node


def _task(**kw):
    base = dict(id="T", phase="p", description="d", prompt_template="t")
    base.update(kw)
    return SubTask(**base)


def test_primary_text_task_gets_prose_not_json():
    """primary + text -> primary_complete."""
    llm = FakeLLM()
    text, parsed = asyncio.run(
        _runner(llm)._llm_complete(_task(model_preference="primary"), "p"))
    assert llm.calls == ["primary_text"]
    assert isinstance(text, str) and text.startswith("## Empfehlung")
    assert parsed == {}


def test_harvest_text_unchanged():
    llm = FakeLLM()
    text, parsed = asyncio.run(
        _runner(llm)._llm_complete(_task(), "p"))
    assert llm.calls == ["harvest_text"]
    assert text == "Kurze Bewertung." and parsed == {}


def test_json_format_keeps_output_a_string():
    """Even with output_format='json', `output` stays a string."""
    llm = FakeLLM()
    text, parsed = asyncio.run(_runner(llm)._llm_complete(
        _task(model_preference="primary", output_format="json"), "p"))
    assert llm.calls == ["primary_json"]
    assert isinstance(text, str)
    assert json.loads(text) == {"empfehlung": "X"}
    assert parsed == {"empfehlung": "X"}


def test_json_on_harvest_model_is_possible():
    """Format and model can be combined freely."""
    llm = FakeLLM()
    text, parsed = asyncio.run(_runner(llm)._llm_complete(
        _task(output_format="json"), "p"))
    assert llm.calls == ["harvest_json"]
    assert parsed == {"score": 3}


# ── Report: no repr, not even with broken output ───────

def _ctx_with(rec_output):
    plan = TaskPlan(use_case="decision_analysis", tasks=[
        SubTask(id="CELL", phase="c", description="Zelle",
                prompt_template="t"),
        SubTask(id="REC", phase="r", description="Empfehlung",
                prompt_template="t", depends_on=["CELL"],
                model_preference="primary"),
    ])

    class Ctx:
        task_plan = plan
        query = "Testanfrage"
        preflight_data = {"decision": "KI-Strategiekonzept"}
        task_results = {
            "CELL": TaskResult(task_id="CELL", state=TaskState.DONE,
                               output="Bewertung: positiv."),
            "REC": TaskResult(task_id="REC", state=TaskState.DONE,
                              output=rec_output),
        }
    return Ctx()


def test_report_has_no_python_repr_even_if_dict_slips_through():
    report = AnalysisPipelineRunner._build_final_report(_ctx_with(
        {"empfehlung": "Lokale Infrastruktur",
         "begruendung": ["Datenhoheit", "Fixkosten"]}))
    assert "'empfehlung'" not in report, "Python repr in the report"
    assert "{'" not in report
    assert "Datenhoheit" in report, "the content must be kept"


def test_report_normal_case_unchanged():
    report = AnalysisPipelineRunner._build_final_report(
        _ctx_with("## Empfehlung\n\nLokale Infrastruktur."))
    assert report.startswith("# KI-Strategiekonzept")
    assert "## Empfehlung" in report


# ── _resolve_prompt: no TypeError at the join ────────────────

def test_concat_deps_survive_a_dict_output():
    """Guards against TypeError: sequence item 0: expected str, dict found."""
    task = SubTask(id="REC", phase="r", description="d",
                   prompt_template="Matrix:\n{matrix}",
                   concat_param_deps={"matrix": ["A", "B"]},
                   depends_on=["A", "B"])
    results = {
        "A": TaskResult(task_id="A", state=TaskState.DONE,
                        output={"score": 1}),      # foreign body
        "B": TaskResult(task_id="B", state=TaskState.DONE,
                        output="Bewertung B"),
    }
    out = AnalysisLayerNode._resolve_prompt(task, results)
    assert "Bewertung B" in out
    assert "'score'" not in out and '"score"' in out


def test_param_dep_with_dict_is_coerced():
    task = SubTask(id="X", phase="p", description="d",
                   prompt_template="Kontext: {ctx}",
                   param_deps={"ctx": "A"}, depends_on=["A"])
    results = {"A": TaskResult(task_id="A", state=TaskState.DONE,
                               output={"k": "v"})}
    out = AnalysisLayerNode._resolve_prompt(task, results)
    assert "'k'" not in out and '"k"' in out


# ── All registered tasks run on text ──────────────────────

def test_json_prompts_and_output_format_agree():
    """Prompt and output format must not diverge.

    Whoever asks for JSON must set output_format='json' — otherwise they get
    prose and parse nothing. And whoever wants prose must not set json.
    Coupling this to `model_preference` would get it wrong.
    """
    pre = {
        "decision_analysis": {"decision": "D", "options": "A\nB",
                              "criteria": "K1\nK2"},
        "peer_review": {"manuscript_summary": "x" * 50, "venue": "J",
                        "focus": "M"},
        "literature_review": {"topic": "T", "discipline": "D",
                              "time_window": "2020", "key_questions": "F"},
        "literature_finder": {"topic": "T", "discipline": "D",
                              "time_window": "2020"},
        "research_design": {"research_question": "R", "field": "F",
                            "constraints": "-"},
        "grant_proposal": {"topic": "T", "funder": "DFG", "duration": "3",
                           "budget": "500k"},
    }

    async def check():
        d = Decomposer.__new__(Decomposer)
        for use_case, params in pre.items():
            plan = await getattr(d, f"_decompose_{use_case}")("Q", params)
            assert plan.tasks, use_case
            for t in plan.tasks:
                asks_json = "json" in t.prompt_template.lower()
                assert t.output_format in ("text", "json"), t.output_format
                if asks_json:
                    assert t.output_format == "json", (
                        f"{use_case}/{t.id} asks for JSON in the prompt, "
                        f"but has output_format='{t.output_format}'"
                    )
                if t.output_format == "json":
                    assert asks_json, (
                        f"{use_case}/{t.id} expects JSON, but does not ask for it in the "
                        f"prompt"
                    )

    asyncio.run(check())
