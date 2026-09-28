"""Tests: "find literature" really queries databases, instead of stopping
after formulating the search queries.
"""

import asyncio
from dataclasses import dataclass, field

import tests.conftest  # noqa: F401  (dependency stubs)

from src.pipeline.analysis_pipeline import (
    AnalysisLayerNode, AnalysisPipelineRunner, Decomposer, _year_from,
)
from src.pipeline.models import TaskResult, TaskState
from src.pipeline.search_executor import (
    MAX_QUERIES, extract_queries, format_results_markdown,
    results_to_dicts, run_literature_search,
)

PREFLIGHT = {"topic": "Hybrides Arbeiten", "discipline": "BWL",
             "time_window": "2020-2025"}


@dataclass
class FakeHit:
    title: str = "Ein Titel"
    authors: list = field(default_factory=lambda: ["Meier", "Schulz"])
    year: int = 2022
    venue: str = "Journal of Work"
    doi: str = "10.1000/abc"
    url: str = "https://example.org/abc"
    abstract: str = "Abstract."
    cited_by_count: int = 12
    sources: list = field(default_factory=lambda: ["openalex"])


class FakeService:
    def __init__(self, per_query=2, fail_on=()):
        self.per_query = per_query
        self.fail_on = set(fail_on)
        self.calls = []

    async def multi_source_query(self, query, **kw):
        self.calls.append((query, kw))
        if query in self.fail_on:
            raise RuntimeError("API down")
        return [FakeHit(title=f"{query} — Paper {i}", doi=f"10.1/{query}{i}",
                        cited_by_count=10 - i)
                for i in range(self.per_query)]


# ── Query extraction ──────────────────────────────────────────────

def test_prefers_structured_queries():
    q = extract_queries("egal", {"queries": ["hybrid work", "remote teams"]})
    assert q == ["hybrid work", "remote teams"]


def test_falls_back_to_lines():
    raw = "1. hybrid work\n- remote teams\n* **telework**\n"
    assert extract_queries(raw, {}) == [
        "hybrid work", "remote teams", "telework"]


def test_skips_headings_and_prose():
    raw = "Suchanfragen:\nhybrid work\n" + "x" * 250
    assert extract_queries(raw, {}) == ["hybrid work"]


def test_query_count_is_capped():
    many = {"queries": [f"q{i}" for i in range(50)]}
    assert len(extract_queries("", many)) == MAX_QUERIES


def test_empty_input_yields_nothing():
    assert extract_queries("", {}) == []
    assert extract_queries(None, None) == []


# ── Execution ────────────────────────────────────────────────────

def test_search_runs_every_query():
    svc = FakeService()
    md, data = asyncio.run(run_literature_search(
        ["hybrid work", "remote teams"], svc, {"limit": 5}))
    assert [c[0] for c in svc.calls] == ["hybrid work", "remote teams"]
    assert svc.calls[0][1]["limit"] == 5
    assert data["n_results"] == 4
    assert "hits" in md


def test_filters_are_passed_through():
    svc = FakeService()
    asyncio.run(run_literature_search(
        ["q"], svc, {"limit": 10, "year_from": 2020, "peer_reviewed_only": True}))
    kw = svc.calls[0][1]
    assert kw["year_from"] == 2020 and kw["peer_reviewed_only"] is True


def test_partial_api_failure_keeps_the_rest():
    """One failure must not break the whole search."""
    svc = FakeService(fail_on=["kaputt"])
    md, data = asyncio.run(run_literature_search(
        ["gut", "kaputt"], svc, {}))
    assert data["n_results"] == 2
    assert data["errors"] and "1 of 2 queries" in md


def test_total_failure_is_reported_clearly():
    svc = FakeService(fail_on=["a", "b"])
    md, data = asyncio.run(run_literature_search(["a", "b"], svc, {}))
    assert data["n_results"] == 0
    assert "failed" in md


def test_no_service_does_not_crash():
    md, data = asyncio.run(run_literature_search(["q"], None, {}))
    assert data["error"] == "no_service" and "not available" in md


def test_no_queries_does_not_call_the_api():
    svc = FakeService()
    md, data = asyncio.run(run_literature_search([], svc, {}))
    assert svc.calls == [] and data["results"] == []


def test_executed_queries_are_shown():
    svc = FakeService()
    md, _ = asyncio.run(run_literature_search(["hybrid work"], svc, {}))
    assert "`hybrid work`" in md


def test_results_are_sorted_by_citations():
    svc = FakeService(per_query=3)
    _, data = asyncio.run(run_literature_search(["q"], svc, {}))
    cites = [r["cited_by_count"] for r in data["results"]]
    assert cites == sorted(cites, reverse=True)


# ── Formatting ──────────────────────────────────────────────────

def test_markdown_contains_verifiable_metadata():
    md = format_results_markdown([FakeHit()])
    for expected in ("Ein Titel", "2022", "Journal of Work",
                     "10.1000/abc", "12 citations", "Meier"):
        assert expected in md, expected


def test_markdown_handles_empty():
    assert "No hits" in format_results_markdown([])


def test_structured_output_keeps_identifiers():
    d = results_to_dicts([FakeHit()])[0]
    assert d["doi"] == "10.1000/abc" and d["year"] == 2022


# ── Time window ───────────────────────────────────────────────────

def test_year_extraction():
    assert _year_from("2020-2025") == 2020
    assert _year_from("seit 2018") == 2018
    assert _year_from("letzte 5 Jahre") is None
    assert _year_from("") is None


# ── The mode as a whole ──────────────────────────────────────────

def _plan():
    d = Decomposer.__new__(Decomposer)
    return asyncio.run(d._decompose_literature_finder("Frage", PREFLIGHT))


def test_mode_has_a_real_search_step():
    plan = _plan()
    ids = [t.id for t in plan.tasks]
    assert ids == ["STRATEGY", "QUERIES", "SEARCH", "ASSESS", "SELECTION"]
    search = next(t for t in plan.tasks if t.id == "SEARCH")
    assert search.executor == "literature_search"


def test_queries_task_is_structured():
    q = next(t for t in _plan().tasks if t.id == "QUERIES")
    assert q.output_format == "json"


def test_time_window_becomes_a_filter():
    search = next(t for t in _plan().tasks if t.id == "SEARCH")
    assert search.search_config["year_from"] == 2020


def test_search_hits_reach_the_report():
    """The hit list is the product and must not be dropped."""
    plan = _plan()
    results = {
        "STRATEGY": TaskResult(task_id="STRATEGY", state=TaskState.DONE,
                               output="Begriffe: hybrid work"),
        "QUERIES": TaskResult(task_id="QUERIES", state=TaskState.DONE,
                              output='{"queries": ["hybrid work"]}',
                              parsed_output={"queries": ["hybrid work"]}),
        "SEARCH": TaskResult(task_id="SEARCH", state=TaskState.DONE,
                             output="**2 Treffer**\n\n1. **Ein Titel** (2022)"),
        "ASSESS": TaskResult(task_id="ASSESS", state=TaskState.DONE,
                             output='{"selected": []}',
                             parsed_output={"selected": []}),
        "SELECTION": TaskResult(task_id="SELECTION", state=TaskState.DONE,
                                output="Bewertung der Treffer."),
    }

    class Ctx:
        task_plan = plan
        query = "Frage"
        preflight_data = PREFLIGHT
        task_results = results

    report = AnalysisPipelineRunner._build_final_report(Ctx())
    assert "Ein Titel" in report, "hit list missing from the report"
    assert "Begriffe: hybrid work" in report, "search strategy missing"
    assert "Bewertung der Treffer." in report
    assert '{"queries"' not in report, "the JSON intermediate step does not belong in it"
    assert "## Literature found" in report


def test_search_task_reads_queries_from_dependency():
    """End to end through the layer node's executor."""
    plan = _plan()
    node = AnalysisLayerNode.__new__(AnalysisLayerNode)
    node._search_service = FakeService()

    class Ctx:
        task_results = {
            "QUERIES": TaskResult(
                task_id="QUERIES", state=TaskState.DONE,
                output='{"queries": ["hybrid work"]}',
                parsed_output={"queries": ["hybrid work"]}),
        }

    search = next(t for t in plan.tasks if t.id == "SEARCH")
    md, data = asyncio.run(node._run_search_task(search, Ctx()))
    assert node._search_service.calls[0][0] == "hybrid work"
    assert data["n_results"] == 2
    assert "hits" in md


def test_other_modes_keep_single_task_reports():
    """No mode except literature_finder may get headings."""
    d = Decomposer.__new__(Decomposer)
    pre = {"decision": "D", "options": "A\nB", "criteria": "K"}
    plan = asyncio.run(d._decompose_decision_analysis("Q", pre))
    assert not any(getattr(t, "show_in_report", False) for t in plan.tasks)


# ── Integration with the real data types ─────────────────────────

def test_works_with_real_searchresult_objects():
    """De-duplication and formatting against the real type, not just stubs."""
    from src.connectors.literature_search import SearchResult

    class RealService:
        async def multi_source_query(self, query, **kw):
            return [
                SearchResult(id="1", doi="10.1/a", title="Hybrid Work Study",
                             authors=["Meier"], year=2022, venue="JOW",
                             cited_by_count=30, sources=["openalex"]),
                # duplicate with the same DOI from another source
                SearchResult(id="2", doi="10.1/a", title="Hybrid Work Study",
                             authors=["Meier"], year=2022, venue="JOW",
                             cited_by_count=30, sources=["s2"]),
                SearchResult(id="3", doi="10.1/b", title="Remote Teams",
                             authors=["Schulz"], year=2021,
                             cited_by_count=5, sources=["arxiv"]),
            ]

    md, data = asyncio.run(run_literature_search(
        ["hybrid work"], RealService(), {}))
    assert data["n_results"] == 2, "duplicate was not merged"
    assert "Hybrid Work Study" in md and "Remote Teams" in md
    assert data["results"][0]["cited_by_count"] == 30, "not sorted"


# ── End to end through the real runner ──────────────────────────

class _E2ELLM:
    """Answers plausibly; `queries` controls the query task."""

    def __init__(self, queries=None):
        self.queries = {"queries": ["hybrid work"]} if queries is None else queries

    async def primary_complete(self, m, **k):
        return "### Bewertung\n\n1. **Ein Titel** — einschlägig."

    async def harvest_complete(self, m, **k):
        return "Suchbegriffe: hybrid work\nEinschluss: peer-reviewed"

    async def primary_complete_json(self, m, **k):
        return self.queries

    async def harvest_complete_json(self, m, **k):
        return self.queries


class _DeadSearch:
    async def multi_source_query(self, query, **kw):
        raise RuntimeError("API unreachable")


class _EmptySearch:
    def __init__(self):
        self.calls = []

    async def multi_source_query(self, query, **kw):
        self.calls.append(query)
        return []


def _run(llm, service):
    from src.pipeline.models import HarvestContext

    checker = __import__(
        "src.pipeline.analysis_pipeline", fromlist=["USE_CASE_REGISTRY"]
    ).USE_CASE_REGISTRY["literature_finder"]["preflight"]
    runner = AnalysisPipelineRunner(
        llm_client=llm,
        progress_callback=lambda *a: asyncio.sleep(0),
        search_service=service,
    )
    ctx = HarvestContext(query=PREFLIGHT["topic"], use_case="literature_finder")
    ctx.preflight_data = checker.normalize(PREFLIGHT)

    async def go():
        c = await runner.decompose_only(ctx)
        return await runner.run(c, skip_decompose=True)

    return asyncio.run(go())


def test_one_input_yields_a_complete_report():
    """The full path: input in, analysed hits out."""
    search = FakeService()
    ctx = _run(_E2ELLM(), search)

    states = {t.id: ctx.task_results[t.id].state.value
              for t in ctx.task_plan.tasks}
    assert states == {"STRATEGY": "done", "QUERIES": "done",
                      "SEARCH": "done", "ASSESS": "done",
                      "SELECTION": "done"}, states
    assert search.calls, "no database was queried"

    report = ctx.final_report
    assert "## Search strategy" in report
    assert "## Literature found" in report
    assert "## Assessment of the hits" in report
    assert "10.1/" in report, "no DOI in the report"
    assert '{"queries"' not in report


def test_runner_accepts_an_injected_search_service():
    """Without an entry point, the search could neither be tested nor share
    an existing API client."""
    search = FakeService()
    runner = AnalysisPipelineRunner(
        llm_client=_E2ELLM(), progress_callback=None,
        search_service=search,
    )
    assert runner.search_service is search


def test_dead_apis_fail_loudly_and_skip_assessment():
    """A failure must not pass as a success.

    Otherwise the assessment task gets an empty hit list — and invents
    plausible-sounding titles from it.
    """
    ctx = _run(_E2ELLM(), _DeadSearch())
    assert ctx.task_results["SEARCH"].state.value == "failed"
    assert ctx.task_results["ASSESS"].state.value == "skipped"
    report = ctx.final_report
    assert "⚠️" in report and "failed" in report
    assert "Assessment of the hits" not in report.split("⚠️")[0]


def test_zero_hits_is_a_valid_result_not_a_failure():
    """A search that ran without hits is not an error."""
    ctx = _run(_E2ELLM(), _EmptySearch())
    assert ctx.task_results["SEARCH"].state.value == "done"
    assert ctx.task_results["ASSESS"].state.value == "done"
    assert "No hits" in ctx.final_report


def test_broken_query_task_falls_back_to_the_topic():
    """Empty or wrongly structured JSON must not break the run."""
    for broken in ({}, {"quatsch": 1}, {"queries": []}):
        search = _EmptySearch()
        ctx = _run(_E2ELLM(queries=broken), search)
        assert search.calls == [PREFLIGHT["topic"]], broken
        assert ctx.task_results["SEARCH"].state.value == "done"


def test_json_text_is_never_line_parsed_into_queries():
    """Otherwise `"quatsch": 1` would reach the API as a search query."""
    assert extract_queries('{\n  "quatsch": 1\n}', {}) == []
    assert extract_queries('[1, 2]', {}) == []
    assert extract_queries("hybrid work\nremote teams", {}) == [
        "hybrid work", "remote teams"]


# ── Citation-Chasing ──────────────────────────────────────────────

class _ChasingService(FakeService):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seeds = None
        self.max_per_seed = None

    async def multi_source_query(self, query, **kw):
        from src.connectors.literature_search import SearchResult
        self.calls.append((query, kw))
        return [
            SearchResult(id="a", doi="10.1/a", title="Viel zitiert",
                         year=2022, cited_by_count=99, sources=["openalex"]),
            SearchResult(id="b", doi="10.1/b", title="Wenig zitiert",
                         year=2021, cited_by_count=3, sources=["s2"]),
            SearchResult(id="c", doi="", title="Ohne DOI",
                         year=2020, cited_by_count=50, sources=["arxiv"]),
        ]

    async def expand_from_seeds(self, seed_dois, max_per_seed=20,
                                directions=None):
        from src.connectors.literature_search import SearchResult
        self.seeds = list(seed_dois)
        self.max_per_seed = max_per_seed
        return [SearchResult(id="z", doi="10.9/z", title="Zitierte Vorarbeit",
                             year=2018, cited_by_count=120,
                             sources=["openalex"])]


def test_citation_chasing_only_on_request():
    svc = _ChasingService()
    asyncio.run(run_literature_search(["q"], svc, {}))
    assert svc.seeds is None, "chasing ran without being asked — it costs 30-90s"


def test_citation_chasing_uses_most_cited_dois_as_seeds():
    svc = _ChasingService()
    _, data = asyncio.run(run_literature_search(
        ["q"], svc, {"expand_citations": True, "max_seeds": 2,
                     "max_per_seed": 7}))
    assert svc.seeds == ["10.1/a", "10.1/b"], svc.seeds
    assert svc.max_per_seed == 7
    assert data["n_expanded"] == 1
    assert any(r["title"] == "Zitierte Vorarbeit" for r in data["results"])


def test_seed_selection_skips_entries_without_doi():
    """A hit without a DOI is useless as a seed."""
    svc = _ChasingService()
    asyncio.run(run_literature_search(
        ["q"], svc, {"expand_citations": True, "max_seeds": 5}))
    assert "Ohne DOI" not in str(svc.seeds)
    assert len(svc.seeds) == 2


def test_failed_chasing_keeps_the_query_results():
    class Broken(_ChasingService):
        async def expand_from_seeds(self, *a, **kw):
            raise RuntimeError("Rate limit")

    svc = Broken()
    md, data = asyncio.run(run_literature_search(
        ["q"], svc, {"expand_citations": True}))
    assert data["n_results"] == 3, "query hits were lost"
    assert any("citation_chasing" in e for e in data["errors"])


def test_chasing_is_reported_in_the_markdown():
    svc = _ChasingService()
    md, _ = asyncio.run(run_literature_search(
        ["q"], svc, {"expand_citations": True}))
    assert "citation chasing" in md


# ── Literature review and grant proposal use the search ───────────────────────────────────

def _plan_for(use_case, preflight):
    d = Decomposer.__new__(Decomposer)
    return asyncio.run(getattr(d, f"_decompose_{use_case}")("Q", preflight))


REVIEW_PRE = {"topic": "Hybrides Arbeiten", "discipline": "BWL",
              "time_window": "2020-2025",
              "key_questions": "Wie definiert?\nWelche Debatten?"}
GRANT_PRE = {"research_question": "Wie wirkt X?", "funder": "DFG",
             "duration": "3 Jahre", "discipline": "BWL"}


def test_literature_review_searches_before_synthesising():
    """A review without literature read is just the model's memory."""
    plan = _plan_for("literature_review", REVIEW_PRE)
    ids = [t.id for t in plan.tasks]
    assert "SEARCH" in ids and "QUERIES" in ids
    search = next(t for t in plan.tasks if t.id == "SEARCH")
    assert search.executor == "literature_search"
    assert search.search_config["expand_citations"] is True
    # every question synthesis must see the hits
    for t in plan.tasks:
        if t.id.startswith("Q_"):
            assert t.param_deps.get("hits") == "SEARCH", t.id
            assert "SEARCH" in t.depends_on


def test_review_questions_may_now_cite_sources():
    """The prompt may ask for titles now that real literature is available."""
    plan = _plan_for("literature_review", REVIEW_PRE)
    q = next(t for t in plan.tasks if t.id.startswith("Q_"))
    assert "{hits}" in q.prompt_template
    assert "not in the list" in q.prompt_template


def test_grant_proposal_backs_the_state_of_the_art_with_literature():
    plan = _plan_for("grant_proposal", GRANT_PRE)
    ids = [t.id for t in plan.tasks]
    assert "SEARCH" in ids
    background = next(t for t in plan.tasks if t.id == "SEC_background")
    assert background.param_deps.get("hits") == "SEARCH"
    # the work plan needs no literature
    workplan = next(t for t in plan.tasks if t.id == "SEC_workplan")
    assert "hits" not in workplan.param_deps


def test_grant_proposal_sections_reach_the_report():
    """The report must contain the proposal sections, not only the coherence check."""
    plan = _plan_for("grant_proposal", GRANT_PRE)
    shown = [t.id for t in plan.tasks if t.show_in_report]
    for key in ("abstract", "background", "methodology", "workplan", "impact"):
        assert f"SEC_{key}" in shown, key
    assert "SEARCH" in shown


def test_report_order_puts_the_proposal_before_the_bibliography():
    plan = _plan_for("grant_proposal", GRANT_PRE)
    order = [t.id for t in plan.tasks if t.show_in_report]
    assert order.index("SEC_impact") < order.index("SEARCH")


def test_both_modes_stay_acyclic_and_resolvable():
    for uc, pre in (("literature_review", REVIEW_PRE),
                    ("grant_proposal", GRANT_PRE)):
        plan = _plan_for(uc, pre)
        layers = plan.topological_order()          # raises on cycles
        flat = [t.id for layer in layers for t in layer]
        assert len(flat) == len(plan.tasks), uc


# ── Selection with complete bibliographic details ───────────

from src.pipeline.search_executor import (  # noqa: E402
    render_selected_references,
)

HITS = [
    {"title": "AI-Powered Educational Agents", "doi": "10.3390/info16060469",
     "authors": ["Córdova-Esparza", "Terven", "Ramírez"], "year": 2025,
     "venue": "Information", "cited_by_count": 59, "sources": ["openalex"],
     "url": ""},
    {"title": "Guidelines and Policies", "doi": "10.1234/hei.2025.7",
     "authors": ["Weber"], "year": 2025, "venue": "Higher Education Policy",
     "cited_by_count": 31, "sources": ["openalex", "s2"], "url": ""},
]


def test_selection_carries_authors_venue_and_doi():
    """The selection contains full details, not only title and year."""
    md = render_selected_references(
        {"selected": [{"doi": "10.3390/info16060469", "reason": "Passt."}]},
        HITS)
    assert "Córdova-Esparza" in md
    assert "*Information*" in md
    assert "10.3390/info16060469" in md
    assert "59 citations" in md
    assert "→ Passt." in md


def test_metadata_comes_from_the_api_not_the_model():
    """Wrong details from the model must not come through."""
    md = render_selected_references(
        {"selected": [{"doi": "10.3390/info16060469",
                       "title": "Falscher Titel vom Modell",
                       "authors": ["Erfundener Autor"], "year": 1999,
                       "reason": "x"}]},
        HITS)
    assert "Falscher Titel" not in md and "Erfundener Autor" not in md
    assert "AI-Powered Educational Agents" in md and "2025" in md


def test_invented_dois_are_dropped_and_counted():
    md = render_selected_references(
        {"selected": [{"doi": "10.3390/info16060469", "reason": "ok"},
                      {"doi": "10.9999/erfunden", "reason": "halluziniert"}]},
        HITS)
    assert "10.9999" not in md
    assert "1 entry was discarded" in md


def test_selection_falls_back_to_title_match():
    """Without a DOI, but with an exact title, the hit stays usable."""
    md = render_selected_references(
        {"selected": [{"title": "Guidelines and Policies", "reason": "ok"}]},
        HITS)
    assert "Higher Education Policy" in md


def test_gaps_and_next_round_are_kept():
    md = render_selected_references(
        {"selected": [], "gaps": "Keine deutschsprachigen Quellen.",
         "next_round": "OECD durchsuchen."}, HITS)
    assert "Gaps in the hits" in md
    assert "Keine deutschsprachigen Quellen." in md
    assert "next search round" in md and "OECD durchsuchen." in md


def test_empty_selection_is_stated_not_faked():
    md = render_selected_references({}, HITS)
    assert "No selection made" in md


def test_finder_plan_renders_the_selection_deterministically():
    plan = _plan()
    ids = [t.id for t in plan.tasks]
    assert ids == ["STRATEGY", "QUERIES", "SEARCH", "ASSESS", "SELECTION"]
    assess = next(t for t in plan.tasks if t.id == "ASSESS")
    assert assess.output_format == "json"
    sel = next(t for t in plan.tasks if t.id == "SELECTION")
    assert sel.executor == "render_selection"
    # the JSON intermediate step must not end up in the report
    shown = [t.id for t in plan.tasks
             if t.show_in_report or not any(t.id in x.depends_on
                                            for x in plan.tasks)]
    assert "ASSESS" not in shown and "SELECTION" in shown
