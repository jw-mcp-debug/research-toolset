"""Reasoning models that use up the whole token budget.

Reproduces a Kimi-like reasoning stream: the model delivers only
`reasoning_content`, uses up the whole token budget with it and ends the
stream with finish_reason="length" — without ever sending visible text.
"""
from src.llm.client import THINKING_PLACEHOLDER
import asyncio

import types

import tests.conftest  # noqa: F401  (dependency stubs)

from src.config import LLMConfig
from src.llm.client import (
    EmptyLLMResponseError, LLMClient, _reasoning_of,
)

def check(name, cond, detail=""):
    assert cond, f"{name}" + (f" [{detail}]" if detail else "")


# ── Fake objects that mimic the OpenAI response structure ────────

class Delta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class Choice:
    def __init__(self, delta=None, message=None, finish_reason=None):
        self.delta = delta
        self.message = message
        self.finish_reason = finish_reason


class Chunk:
    def __init__(self, choice):
        self.choices = [choice]


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c
        return gen()

    async def close(self):
        pass


class FakeCompletions:
    """Record the max_tokens of every call (for the escalation)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs.get("max_tokens"))
        return self.script.pop(0)


def wire(client, script):
    comp = FakeCompletions(script)
    client.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=comp)
    )
    return comp


def make_client(max_output_tokens=32768):
    cfg = LLMConfig(model_name="kimi-25",
                    max_output_tokens=max_output_tokens)
    return LLMClient(cfg)


async def collect(gen):
    out = []
    async for chunk in gen:
        out.append(chunk)
    return out


# ── 1. The reasoning field is recognised at all ─────────────────────

def test_reasoning_field_detection():
    check("_reasoning_of reads reasoning_content",
          _reasoning_of(Delta(reasoning_content="denk denk")) == "denk denk")
    check("_reasoning_of ignores plain content",
          _reasoning_of(Delta(content="Text")) == "")


# ── 2. A stream with reasoning only ────────────────────

async def test_stream_reasoning_only():
    client = make_client()
    # budget exhausted -> every attempt ends the same way
    def budget_burn():
        return FakeStream([
            Chunk(Choice(delta=Delta(reasoning_content="x" * 400)))
            for _ in range(10)
        ] + [Chunk(Choice(delta=Delta(), finish_reason="length"))])

    comp = wire(client, [budget_burn() for _ in range(4)])

    raised = None
    chunks = []
    try:
        async for c in client.stream([{"role": "user", "content": "q"}],
                                     max_tokens=16384):
            chunks.append(c)
    except EmptyLLMResponseError as e:
        raised = e

    check("a stream without visible text raises EmptyLLMResponseError",
          raised is not None,
          f"finish_reason={getattr(raised, 'finish_reason', None)}")
    check("reasoning produces NO UI updates",
          chunks == [], f"{len(chunks)} Chunks ausgegeben")
    check("the budget was escalated once",
          comp.calls[:2] == [16384, 32768], f"calls={comp.calls}")

def test_stream_reasoning_only_sync():
    asyncio.run(test_stream_reasoning_only())


# ── 3. Regression: a normal stream keeps working ────────────

async def test_stream_ok():
    client = make_client()
    wire(client, [FakeStream([
        Chunk(Choice(delta=Delta(reasoning_content="ueberlege..."))),
        Chunk(Choice(delta=Delta(content="# Bericht\n"))),
        Chunk(Choice(delta=Delta(content="Inhalt."),
                     finish_reason="stop")),
    ])])
    chunks = await collect(client.stream([{"role": "user", "content": "q"}]))
    check("reasoning + text -> only the text arrives",
          chunks[-1] == "# Bericht\nInhalt.", repr(chunks[-1]))
    check("no placeholder spam during reasoning",
          not any(THINKING_PLACEHOLDER in c for c in chunks),
          f"{len(chunks)} Chunks")

def test_stream_ok_sync():
    asyncio.run(test_stream_ok())


async def test_unclosed_think_tag():
    """The placeholder from an open <think> block is not content."""
    client = make_client(max_output_tokens=4096)
    def only_think():
        return FakeStream([
            Chunk(Choice(delta=Delta(content="<think>gruebel"))),
            Chunk(Choice(delta=Delta(content=" weiter"),
                         finish_reason="length")),
        ])
    wire(client, [only_think() for _ in range(4)])
    raised = None
    try:
        await collect(client.stream([{"role": "user", "content": "q"}],
                                    max_tokens=4096))
    except EmptyLLMResponseError as e:
        raised = e
    check("an open <think> block does not count as visible text",
          raised is not None)

def test_unclosed_think_tag_sync():
    asyncio.run(test_unclosed_think_tag())


# ── 4. complete(): escalation rescues the map call ─────────────────

async def test_complete_escalation():
    client = make_client()
    empty = types.SimpleNamespace(
        choices=[Choice(message=types.SimpleNamespace(
            content="", reasoning_content="y" * 5000),
            finish_reason="length")],
        usage=None)
    full = types.SimpleNamespace(
        choices=[Choice(message=types.SimpleNamespace(
            content="Die Antwort.", reasoning_content="y" * 200),
            finish_reason="stop")],
        usage=None)
    comp = wire(client, [empty, full])
    out = await client.complete([{"role": "user", "content": "q"}],
                                max_tokens=4096)
    check("complete() escalates the budget and then returns text",
          out == "Die Antwort.", f"calls={comp.calls}")

def test_complete_escalation_sync():
    asyncio.run(test_complete_escalation())


async def test_complete_gives_up_loudly():
    client = make_client(max_output_tokens=4096)  # no leeway
    empty = types.SimpleNamespace(
        choices=[Choice(message=types.SimpleNamespace(
            content="", reasoning_content="y" * 5000),
            finish_reason="length")],
        usage=None)
    wire(client, [empty])
    try:
        await client.complete([{"role": "user", "content": "q"}],
                              max_tokens=4096)
        check("complete() without leeway raises an exception", False)
    except EmptyLLMResponseError:
        check("complete() without leeway raises an exception", True)

def test_complete_gives_up_loudly_sync():
    asyncio.run(test_complete_gives_up_loudly())


# ── 5. Fallback helpers in the orchestrator ────────────────────────────

from src.pipeline.orchestrator import (
    _report_from_map_answers, _strip_thinking_placeholder,
)

def test_orchestrator_fallback_helpers():
    check("the placeholder is discarded as a report",
          _strip_thinking_placeholder(THINKING_PLACEHOLDER) == ""
          and _strip_thinking_placeholder("🤔 *Denkt nach...*") == "")
    check("real text stays untouched",
          _strip_thinking_placeholder("# Bericht\n\nInhalt")
          == "# Bericht\n\nInhalt")

    body = _report_from_map_answers([
        {"question_id": "F1", "question": "Wer ist X?", "answer": "X ist Y."},
        {"question_id": "F2", "question": "Leer?", "answer": "  "},
        {"question_id": "F3", "question": "Rolle?", "answer": "Leitung."},
    ])
    check("the emergency report uses only filled map answers",
          body.count("##") == 2 and "X ist Y." in body, repr(body[:60]))
    check("no pseudo-report without map answers",
          _report_from_map_answers([]) == "")


# ── 6. Node guards ──────────────────────────────────────────────

from src.pipeline.dag_nodes import (
    _MIN_REPORT_CHARS, ReportRevisionNode, ReportWarningBannerNode,
)


class Ctx:
    def __init__(self, **kw):
        self.final_report = ""
        self.factoid_verifications = []
        self.report_quality = None
        self.query_fulfillment = None
        self.synthesis_degraded = ""
        self.__dict__.update(kw)


def test_revision_guards():
    # A degenerate report: 260 characters, 8 contradicted statements
    degenerate = Ctx(
        final_report="# Jonas Brenner\n\n> Anfrage\n\n---\n\n*Erstellt*"
                     + " " * 200,
        factoid_verifications=[
            {"verified": "false", "confidence": 0.95, "factoid": f"A{i}"}
            for i in range(8)
        ],
    )
    ok, reason = ReportRevisionNode(llm=None).applies_to(degenerate)
    check("the revision leaves a degenerate report alone", ok is False, reason)

    gesund = Ctx(final_report="x" * 5000,
                 factoid_verifications=degenerate.factoid_verifications)
    ok, _ = ReportRevisionNode(llm=None).applies_to(gesund)
    check("the revision still runs for a normal report", ok is True)


async def test_banner():
    node = ReportWarningBannerNode()
    ctx = Ctx(
        final_report="# Titel\n\nInhalt",
        report_quality={"passed": False, "rating": "poor",
                        "issues": [{"description": "Keine Fakten"}]},
        query_fulfillment={"fulfilled": False, "assessment": "nur Metadaten",
                           "rework": "Synthese wiederholen"},
        synthesis_degraded="Reasoning-Budget erschoepft.",
    )
    ok, _ = node.applies_to(ctx)
    check("the banner triggers on complaints", ok is True)
    res = await node.run(ctx)
    check("the banner reports all three findings", res["problems"] == 3,
          str(res))
    check("the findings are in the report",
          "Quality check not passed" in ctx.final_report
          and "Reasoning-Budget" in ctx.final_report)

    clean = Ctx(final_report="# Titel\n\nInhalt",
                 report_quality={"passed": True},
                 query_fulfillment={"fulfilled": True})
    ok, reason = node.applies_to(clean)
    check("no banner for a clean run", ok is False, reason)

def test_banner_sync():
    asyncio.run(test_banner())

def test_min_report_threshold():
    check("the threshold covers the degenerate report (260 characters)",
          _MIN_REPORT_CHARS > 260, f"_MIN_REPORT_CHARS={_MIN_REPORT_CHARS}")
