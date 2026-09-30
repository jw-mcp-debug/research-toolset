"""
Data models of the research pipeline.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from src.core.vocab import HIGH, normalize_level


class SourceType(Enum):
    WEB_SEARCH = "web_search"
    WEB_PAGE = "web_page"
    GIT_REPO = "git_repo"
    GIT_FILE = "git_file"
    GIT_ISSUE = "git_issue"
    LOCAL_FILE = "local_file"
    ELASTIC = "elastic"
    DIRECTORY_PERSON = "directory_person"
    DIRECTORY_ORG = "directory_org"


class ResearchPhase(Enum):
    FORMAT = "format"
    PLAN = "plan"
    SEARCH = "search"
    FETCH = "fetch"
    HARVEST = "harvest"
    GAP_ANALYSIS = "gap_analysis"
    SYNTHESIS = "synthesis"
    DONE = "done"
    ERROR = "error"


# ─── Connectors ────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Search result (not yet fetched)."""
    title: str
    url: str
    snippet: str
    source_type: SourceType
    connector_name: str = ""


@dataclass
class SourceDocument:
    """Fetched document from any source."""
    source_type: SourceType
    url: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)
    fetch_time_seconds: float = 0.0
    content_length: int = 0
    # Publication date of the source (ISO 8601, e.g. "2016-09-08").
    # Extracted by the connector from HTML metadata (article:published_time,
    # JSON-LD datePublished, <time datetime>, visible date headers).
    # Empty string if no date was found.
    # Use: anchor for resolving dates in the harvest prompt ("since
    # September" without a year refers to the publication date, not to
    # the current date).
    published_date: str = ""

    def __post_init__(self):
        self.content_length = len(self.content)


# ─── Research plan ─────────────────────────────────────────────────

@dataclass
class ResearchQuestion:
    """A single research question."""
    id: str                     # F1, F2, ...
    question: str
    search_terms: list[str] = field(default_factory=list)
    # Search terms per language: {"de": [...], "en": [...]}
    # Built by the parser from the LLM output (dict or flat list).
    # If empty, the orchestrator falls back to search_terms + search_langs.
    search_terms_by_lang: dict = field(default_factory=dict)
    source_scope: str = "web"   # web, github, gitlab, local, elastic
    priority: str = "high"      # high | medium | low (src.core.vocab)
    search_langs: list[str] = field(default_factory=lambda: ["de", "en"])
    answered: bool = False

    @classmethod
    def from_dict(cls, d: dict, fallback_id: str = "") -> "ResearchQuestion":
        """Read a ResearchQuestion from a dict (typically LLM JSON).

        Defensive: missing fields are set to defaults, wrong types are
        ignored. Always returns a valid instance. If `id` is missing and
        `fallback_id` is given, the latter is used (e.g. "F3" for the
        third question).

        `search_terms` are normalised via `coerce_search_terms`, because
        the LLM may return this field in several formats (flat list, dict
        per language, list of {term, lang} dicts). Result: flat
        `search_terms` + structured `search_terms_by_lang`.
        """
        if not isinstance(d, dict):
            return cls(id=fallback_id, question="")
        # Normalising search_terms belongs in the conversion, because it is
        # specific to LLM output
        from src.pipeline.query_utils import coerce_search_terms
        flat_terms, terms_by_lang = coerce_search_terms(
            d.get("search_terms", [])
        )
        # If the input already contains the per-language form (e.g. from a
        # persisted plan), keep it
        existing_by_lang = d.get("search_terms_by_lang", {})
        if isinstance(existing_by_lang, dict) and existing_by_lang:
            terms_by_lang = {**terms_by_lang, **existing_by_lang}
        langs = d.get("search_langs", ["de", "en"])
        if not isinstance(langs, list):
            langs = ["de", "en"]
        return cls(
            id=str(d.get("id", "") or fallback_id),
            question=str(d.get("question", "") or ""),
            search_terms=flat_terms,
            search_terms_by_lang=terms_by_lang,
            source_scope=str(d.get("source_scope", "web") or "web"),
            priority=normalize_level(d.get("priority"), default=HIGH),
            search_langs=[str(l) for l in langs if str(l).strip()],
            answered=bool(d.get("answered", False)),
        )

    def to_dict(self) -> dict:
        """Serialise the question as a dict (typically for JSON persistence)."""
        return {
            "id": self.id,
            "question": self.question,
            "search_terms": list(self.search_terms),
            "search_terms_by_lang": dict(self.search_terms_by_lang),
            "source_scope": self.source_scope,
            "priority": self.priority,
            "search_langs": list(self.search_langs),
            "answered": self.answered,
        }


@dataclass
class DirectURL:
    """A URL to fetch directly."""
    url: str
    reason: str

    @classmethod
    def from_dict(cls, d: dict) -> "DirectURL":
        if not isinstance(d, dict):
            return cls(url="", reason="")
        return cls(
            url=str(d.get("url", "") or "").strip().rstrip("]}).,:;'\""),
            reason=str(d.get("reason", "") or ""),
        )

    def to_dict(self) -> dict:
        return {"url": self.url, "reason": self.reason}


@dataclass
class GitRepoTarget:
    """A Git repository to search specifically."""
    owner: str
    repo: str
    platform: str = "github"    # github, gitlab
    search_scope: str = "readme"  # readme, code, issues
    search_terms: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "GitRepoTarget":
        if not isinstance(d, dict):
            return cls(owner="", repo="")
        terms = d.get("search_terms", [])
        if not isinstance(terms, list):
            terms = []
        return cls(
            owner=str(d.get("owner", "") or ""),
            repo=str(d.get("repo", "") or ""),
            platform=str(d.get("platform", "github") or "github"),
            search_scope=str(d.get("search_scope", "readme") or "readme"),
            search_terms=[str(t) for t in terms if str(t).strip()],
        )

    def to_dict(self) -> dict:
        return {
            "owner": self.owner,
            "repo": self.repo,
            "platform": self.platform,
            "search_scope": self.search_scope,
            "search_terms": list(self.search_terms),
        }


@dataclass
class DirectoryQuery:
    """A query to the person directory."""
    query_type: str = "search"  # always "search"
    query: str = ""             # search text
    reason: str = ""
    pid: int = 0                # Unused, kept for compatibility
    with_children: bool = True
    with_subobjects: bool = True
    subject_area: str = ""        # optional: filter by subject area
    org_filter: str = ""        # optional: filter by organisation path

    @classmethod
    def from_dict(cls, d: dict) -> "DirectoryQuery":
        """Read a person-directory query (LLM output and persisted form)."""
        if not isinstance(d, dict):
            return cls()
        search_text = str(d.get("query", "") or "")
        return cls(
            query_type=str(d.get("query_type", "search") or "search"),
            query=search_text,
            reason=str(d.get("reason", "") or ""),
            subject_area=str(d.get("subject_area", "") or ""),
            org_filter=str(d.get("org_filter", "") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "query": self.query,
            "reason": self.reason,
            "subject_area": self.subject_area,
            "org_filter": self.org_filter,
        }


@dataclass
class ResearchPlan:
    """Research plan generated by the analysis LLM."""
    summary: str = ""
    questions: list[ResearchQuestion] = field(default_factory=list)
    direct_urls: list[DirectURL] = field(default_factory=list)
    git_repos: list[GitRepoTarget] = field(default_factory=list)
    directory_queries: list[DirectoryQuery] = field(default_factory=list)
    followup_queries: list[dict] = field(default_factory=list)

    def add_followup_queries(self, queries: list[dict]):
        self.followup_queries.extend(queries)

    def get_unanswered_questions(self) -> list[ResearchQuestion]:
        return [q for q in self.questions if not q.answered]

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchPlan":
        """Read a ResearchPlan from a dict (LLM JSON output or a stored plan).

        Defensive: missing or mistyped fields are set to defaults,
        individual invalid entries in lists are discarded. A valid plan is
        ALWAYS returned — even if the input dict is completely broken.

        This method encapsulates the JSON→plan conversion in one place.
        Use cases:
          - LLM output (`primary_complete_json`) → plan
          - persisted plan (run.json) → plan
          - test fixture from a dict → plan

        Anti-pattern to avoid: caller code that copies the LLM dict output
        field by field into `ResearchPlan(...)` — such places should call
        `ResearchPlan.from_dict()`.
        """
        if not isinstance(d, dict):
            return cls()
        questions = []
        for i, q in enumerate(d.get("questions", []) or [], start=1):
            try:
                rq = ResearchQuestion.from_dict(q, fallback_id=f"F{i}")
                # Discard empty questions — with neither text nor terms the input
                # was probably broken
                if rq.question or rq.search_terms:
                    questions.append(rq)
            except Exception:
                continue
        direct_urls = []
        for u in d.get("direct_urls", []) or []:
            try:
                du = DirectURL.from_dict(u)
                if du.url:
                    direct_urls.append(du)
            except Exception:
                continue
        git_repos = []
        for g in d.get("git_repos", []) or []:
            try:
                gr_target = GitRepoTarget.from_dict(g)
                if gr_target.owner and gr_target.repo:
                    git_repos.append(gr_target)
            except Exception:
                continue
        directory_queries = []
        for z in d.get("directory_queries", []) or []:
            try:
                zq = DirectoryQuery.from_dict(z)
                if zq.query:
                    directory_queries.append(zq)
            except Exception:
                continue
        followup = d.get("followup_queries", []) or []
        if not isinstance(followup, list):
            followup = []
        return cls(
            summary=str(d.get("summary", "") or ""),
            questions=questions,
            direct_urls=direct_urls,
            git_repos=git_repos,
            directory_queries=directory_queries,
            followup_queries=followup,
        )

    def to_dict(self) -> dict:
        """Serialise the plan as a dict — JSON-compatible.

        Use cases: persistence in `run.json`, logging, inter-process
        communication, test snapshots.
        """
        return {
            "summary": self.summary,
            "questions": [q.to_dict() for q in self.questions],
            "direct_urls": [du.to_dict() for du in self.direct_urls],
            "git_repos": [gr.to_dict() for gr in self.git_repos],
            "directory_queries": [zq.to_dict() for zq in self.directory_queries],
            "followup_queries": list(self.followup_queries),
        }


# ─── Output-Schema ──────────────────────────────────────────────────

@dataclass
class OutputSchema:
    """Output schema generated by the format agent."""
    format_type: str = "structured_report"
    title: str = ""
    sections: list[dict] = field(default_factory=list)
    per_section_fields: list[str] = field(default_factory=list)
    style: str = "factual, clear and direct"
    language: str = "de"
    synthesis_guidance: str = ""

    @classmethod
    def from_dict(cls, d: dict, fallback_title: str = "") -> "OutputSchema":
        """Read an OutputSchema from a dict (typically LLM JSON from the
        format agent).

        Defensive: missing fields are set to defaults, wrong types are
        ignored. Always returns a valid instance. If `title` is missing
        and `fallback_title` is given, the latter is used.
        """
        if not isinstance(d, dict):
            return cls(title=fallback_title)
        sections = d.get("sections", [])
        if not isinstance(sections, list):
            sections = []
        per_section = d.get("per_section_fields", [])
        if not isinstance(per_section, list):
            per_section = []
        return cls(
            format_type=str(
                d.get("format_type", "structured_report")
                or "structured_report"
            ),
            title=str(d.get("title", "") or fallback_title),
            sections=[s for s in sections if isinstance(s, dict)],
            per_section_fields=[str(p) for p in per_section if str(p).strip()],
            style=str(d.get("style", "factual, clear and direct")
                      or "factual, clear and direct"),
            language=str(d.get("language", "de") or "de"),
            synthesis_guidance=str(d.get("synthesis_guidance", "") or ""),
        )

    def to_dict(self) -> dict:
        """Serialise as a dict (for JSON persistence, logging)."""
        return {
            "format_type": self.format_type,
            "title": self.title,
            "sections": list(self.sections),
            "per_section_fields": list(self.per_section_fields),
            "style": self.style,
            "language": self.language,
            "synthesis_guidance": self.synthesis_guidance,
        }


# ─── Extracts ───────────────────────────────────────────────────────

@dataclass
class SourceExtract:
    """Extracts from a single source.

    The `polarity` field is set directly by the harvest LLM, instead of
    guessing negative extracts with a phrase list. Three classes:

      - "positive": answers the question with concrete content → goes
                    into the report.
      - "negative": states the absence of information ("the source does
                    not cover topic X in detail"). Hidden in the
                    synthesis, but counts for the coverage assessment.
      - "meta":     a statement about the source situation itself ("the
                    source is marketing material, not a datasheet").
                    Collected in a separate methodology section of the
                    report.

    The default is "positive": extracts without a marker are treated as
    positive. That is conservative — an unmarked negative extract then
    reaches the report, which is less harmful than the opposite case.
    """
    source_url: str
    source_title: str
    question_id: str            # F1, F2, ...
    fact: str
    context: str = ""
    reliability: str = "medium"  # high | medium | low (src.core.vocab)
    polarity: str = "positive"   # positive | negative | meta


@dataclass
class HarvestResult:
    """Result of the harvest phase for a single source."""
    source_url: str
    source_title: str
    extracts: list[SourceExtract] = field(default_factory=list)
    is_relevant: bool = True
    raw_response: str = ""


# ─── Gap analysis ────────────────────────────────────────────────────

@dataclass
class GapAnalysis:
    """Result of the gap analysis."""
    should_continue: bool = False
    reasoning: str = ""
    new_queries: list[dict] = field(default_factory=list)
    answered_questions: list[str] = field(default_factory=list)
    direct_urls: list[dict] = field(default_factory=list)


# ─── Progress ────────────────────────────────────────────────────

@dataclass
class ProgressUpdate:
    """A progress update entry."""
    phase: "ResearchPhase"
    message: str
    detail: str = ""
    progress_percent: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE: data models for multi-stage analysis workflows
# ═══════════════════════════════════════════════════════════════════

class TaskState(Enum):
    """State of a sub-task in the DAG executor."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubTask:
    """A single sub-task within an analysis pipeline.

    Sub-tasks are the units the DAG executor runs. They have a unique ID,
    a phase, a prompt name, parameters and dependencies on other
    sub-tasks.
    """
    id: str                                       # e.g. "B3", "explanation_concept_5"
    phase: str                                    # e.g. "concept_explanation"
    description: str                              # human-readable description
    prompt_template: str                          # name of the prompt template
    prompt_params: dict = field(default_factory=dict)  # parameters for the template
    depends_on: list[str] = field(default_factory=list)  # IDs of other SubTasks
    param_deps: dict = field(default_factory=dict)       # template_var → task_id (for dynamic values)
    concat_param_deps: dict = field(default_factory=dict)  # template_var → list[task_id] (concatenated)
    parallelizable: bool = True                   # may run in parallel with its peers
    estimated_tokens: int = 2000                  # for progress estimation
    max_retries: int = 2                          # number of retries
    model_preference: str = "harvest"             # "primary" | "harvest"
    # Expected output form: "text" (prose/Markdown) or "json".
    #
    # Deliberately separate from `model_preference`: which model computes
    # and which form the answer has are two different questions. Coupling
    # them would force prose prompts into a JSON corset (every task with
    # model_preference="primary" getting a JSON call) and put the parsed
    # dict into the report as a Python repr.
    #
    # With "json" the serialised text goes into `TaskResult.output` and
    # the parsed structure additionally into `TaskResult.parsed_output`
    # — so `output` remains a string under all circumstances.
    output_format: str = "text"                   # "text" | "json"
    # Who runs the task?
    #
    # "llm"                → prompt to a language model (default)
    # "literature_search"  → a real query of the literature APIs
    #                        (OpenAlex, Semantic Scholar, arXiv) via
    #                        `LiteratureSearchService`
    #
    # Without this, the analysis path could only formulate, never
    # retrieve: "find literature" would propose search terms without ever
    # asking a search engine.
    # "render_selection" joins a model's selection with the structured
    # hits of a search task — without an LLM call, so that bibliographic
    # details come from the API and are not copied by the model.
    executor: str = "llm"   # "llm" | "literature_search" | "render_selection"
    # Parameters for non-LLM executors, e.g.
    # {"queries_from": "QUERIES", "limit": 25, "year_from": 2020}
    search_config: dict = field(default_factory=dict)
    # Include in the report even if other tasks depend on this one.
    # Otherwise `_build_final_report` only includes tasks nothing depends
    # on — with a literature search the hit list itself would then be
    # invisible.
    show_in_report: bool = False
    use_thinking: bool = False                    # enable thinking mode


@dataclass
class TaskResult:
    """Result of an executed sub-task."""
    task_id: str
    state: TaskState = TaskState.PENDING
    output: str = ""                              # raw LLM answer
    parsed_output: dict = field(default_factory=dict)  # structured (JSON parse)
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    tokens_used: int = 0
    retries_used: int = 0

    def to_dict(self) -> dict:
        """Serialisable."""
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "output": self.output,
            "parsed_output": self.parsed_output,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "tokens_used": self.tokens_used,
            "retries_used": self.retries_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskResult":
        """Deserialise from a dict."""
        return cls(
            task_id=data["task_id"],
            state=TaskState(data["state"]),
            output=data.get("output", ""),
            parsed_output=data.get("parsed_output", {}),
            error=data.get("error", ""),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            tokens_used=data.get("tokens_used", 0),
            retries_used=data.get("retries_used", 0),
        )


@dataclass
class TaskPlan:
    """Decomposed execution plan of an analysis pipeline.

    Contains the sub-tasks as a DAG (via the depends_on fields). The
    method topological_order() returns a list of layers (lists of tasks)
    that can run in parallel.
    """
    use_case: str                                 # "explainer", "review", ...
    tasks: list[SubTask] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    estimated_calls: int = 0
    estimated_duration_seconds: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.estimated_calls:
            self.estimated_calls = len(self.tasks)

    def topological_order(self) -> list[list[SubTask]]:
        """Return layers of tasks that can run in parallel.

        Uses Kahn's algorithm: tasks without open dependencies go into the
        next layer. Raises ValueError on cycles.
        """
        task_by_id = {t.id: t for t in self.tasks}

        # Validation: all depends_on must exist
        for task in self.tasks:
            for dep in task.depends_on:
                if dep not in task_by_id:
                    raise ValueError(
                        f"Task '{task.id}' depends on non-existent "
                        f"task '{dep}'"
                    )

        # Count incoming edges
        in_degree = {t.id: len(t.depends_on) for t in self.tasks}
        layers: list[list[SubTask]] = []
        remaining = set(in_degree.keys())

        while remaining:
            # Current layer: all tasks without open dependencies
            current_layer = [
                task_by_id[tid] for tid in remaining
                if in_degree[tid] == 0
            ]
            if not current_layer:
                # Cycle!
                raise ValueError(
                    f"Cycle detected in the TaskPlan. Remaining tasks: "
                    f"{remaining}"
                )

            layers.append(current_layer)
            for task in current_layer:
                remaining.discard(task.id)
                # Reduce the edges to successors
                for other_id in remaining:
                    if task.id in task_by_id[other_id].depends_on:
                        in_degree[other_id] -= 1

        return layers

    def to_preview_markdown(self) -> str:
        """Human-readable plan overview for the UI."""
        try:
            layers = self.topological_order()
        except ValueError as e:
            return f"⚠️ Invalid plan: {e}"

        lines = [
            f"## Execution plan: {self.use_case}",
            "",
            f"**Total:** {len(self.tasks)} sub-tasks in "
            f"{len(layers)} layers",
            f"**Estimated calls:** {self.estimated_calls}",
            f"**Estimated duration:** ~{self.estimated_duration_seconds}s",
            "",
        ]

        for i, layer in enumerate(layers, 1):
            par_note = " (parallel)" if len(layer) > 1 else ""
            lines.append(f"### Layer {i}{par_note}")
            for task in layer:
                lines.append(f"- **{task.id}** [{task.phase}]: {task.description}")
            lines.append("")

        return "\n".join(lines)




@dataclass
class ExecutionCheckpoint:
    """Persisted pipeline state between layers.

    Saved after every successful layer. Allows resuming after a crash,
    a browser reload or a user abort.
    """
    plan_id: str                                  # unique plan ID
    use_case: str
    completed_layers: list[int] = field(default_factory=list)
    task_results: dict[str, dict] = field(default_factory=dict)  # serialised
    running_context_text: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "use_case": self.use_case,
            "completed_layers": self.completed_layers,
            "task_results": self.task_results,
            "running_context_text": self.running_context_text,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionCheckpoint":
        return cls(
            plan_id=data["plan_id"],
            use_case=data["use_case"],
            completed_layers=data.get("completed_layers", []),
            task_results=data.get("task_results", {}),
            running_context_text=data.get("running_context_text", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class HarvestContext:
    """The whole context of a research run — serialisable."""
    id: str = ""
    query: str = ""
    # Output language code for the report and exports (src.output_language).
    output_language: str = "en"
    chat_history: list[dict] = field(default_factory=list)
    output_schema: Optional[OutputSchema] = None
    research_plan: Optional[ResearchPlan] = None
    sources: list[SourceDocument] = field(default_factory=list)
    extracts: list[SourceExtract] = field(default_factory=list)
    harvest_results: list[HarvestResult] = field(default_factory=list)
    gap_analyses: list[GapAnalysis] = field(default_factory=list)
    progress_log: list[ProgressUpdate] = field(default_factory=list)
    final_report: str = ""
    rounds_completed: int = 0
    started_at: str = ""
    finished_at: str = ""
    status: str = "pending"     # pending, running, done, error
    error_message: str = ""
    # Research strategy metadata
    search_stats: dict = field(default_factory=dict)  # langs, terms_by_lang, etc.
    # Person-directory verification
    directory_verification_warnings: list[dict] = field(default_factory=list)
    # LLM usage statistics (transparency)
    llm_usage: dict = field(default_factory=dict)
    # Contradictions between extracts
    contradiction_warnings: list[dict] = field(default_factory=list)

    # ── Classifier layer ──
    # Per round we collect the coverage assessments and the continue
    # decision, instead of mechanical thresholds ("high_rel >= 2",
    # "round_number >= 2 → unanswerable").
    # Format per entry in coverage_per_round:
    #     {"F1": {"coverage": "answered", "confidence": 0.92,
    #             "missing_aspects": [...], "supporting_extract_ids": [...],
    #             "fallback_used": False}, ...}
    coverage_per_round: list[dict] = field(default_factory=list)
    continue_decisions: list[dict] = field(default_factory=list)
    # Diagnosis result at the end of the run; None until the diagnosis ran
    final_diagnosis: Optional[dict] = None
    # Filter statistics per round (visibility)
    filter_stats_per_round: list[dict] = field(default_factory=list)
    # Factoid verification
    # Per entry: {"factoid": str, "type": str, "verified": str,
    #               "supporting_extract_id": str|None,
    #               "supporting_quote": str, "confidence": float}
    factoid_verifications: list[dict] = field(default_factory=list)
    # Report quality — set by ReportQualityNode. Structure: {passed: bool,
    # rating: str, issues: list[dict], segments: int,
    # fallback_used: bool}
    report_quality: Optional[dict] = None
    # Fulfilment of the original user query — set by QueryFulfillmentNode.
    # Structure: {fulfilled: bool, assessment: str, rework: str,
    # fallback_used: bool}
    query_fulfillment: Optional[dict] = None

    # ── Synthesis map answers (map + reduce) ──
    # Per research question the condensed answer from the map phase.
    # Structure per entry: {"question_id": str, "question": str,
    #                        "answer": str, "n_extracts": int}
    # Shown in the pipeline-run tab to make the intermediate synthesis
    # step visible.
    map_answers: list[dict] = field(default_factory=list)

    # ── Degraded synthesis ──
    # Empty string = all normal. Otherwise the reason why the report does
    # not come from the regular reduce phase (e.g. reasoning budget used
    # up → emergency report from the map answers).
    # Shown in the report header and the warning banner.
    synthesis_degraded: str = ""

    # ── Report revision (set by ReportRevisionNode) ──
    # If the factoid verifier found contradicted statements with high
    # confidence and the report was rewritten:
    # what was changed?
    # Structure:
    #   {"applied": bool, "n_corrected": int,
    #    "factoids_corrected": list[str],
    #    "original_length": int, "corrected_length": int,
    #    "reason": str (only if applied=False)}
    report_revision: Optional[dict] = None

    # ── Classifier calls (transparency) ──
    # Collects all ClassifierCall entries of the run. Structure per
    # entry see ClassifierCall.to_dict() — at least:
    #     {"name": str, "input_summary": dict, "output": dict,
    #      "fallback_used": bool, "duration_ms": int}
    classifier_calls: list = field(default_factory=list)

    # ── Analysis pipeline (for analysis/construction modes) ──
    use_case: str = ""                            # "explainer", "review", ...
    task_plan: Optional[TaskPlan] = None
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    running_context_text: str = ""                # accumulated context
    preflight_data: dict = field(default_factory=dict)  # validated mandatory inputs
    user_edits: dict = field(default_factory=dict)      # manual plan adjustments
    plan_confirmed: bool = False                  # the user confirmed the plan in the plan gate
    completed_layers: list[int] = field(default_factory=list)  # for resume

    def __post_init__(self):
        if not self.id:
            # Microsecond precision + a short random suffix
            # prevent collisions between simultaneous pipeline starts
            import secrets
            now = datetime.now()
            suffix = secrets.token_hex(3)  # 6 hex characters
            self.id = f"{now.strftime('%Y%m%d_%H%M%S_%f')}_{suffix}"
        if not self.started_at:
            self.started_at = datetime.now().isoformat()

    @property
    def duration_seconds(self) -> float:
        if not self.finished_at:
            return (datetime.now() - datetime.fromisoformat(self.started_at)).total_seconds()
        return (datetime.fromisoformat(self.finished_at) -
                datetime.fromisoformat(self.started_at)).total_seconds()

    @property
    def unique_source_urls(self) -> set[str]:
        return {s.url for s in self.sources}

    def log_progress(self, phase: ResearchPhase, message: str,
                     detail: str = "", progress: float = 0.0):
        self.progress_log.append(ProgressUpdate(
            phase=phase, message=message,
            detail=detail, progress_percent=progress,
        ))

    def save(self, data_dir: str):
        """Save the whole context to disk."""
        # Directory name from the timestamp + a short form of the request
        slug = self.query[:50].replace(" ", "-").replace("/", "_")
        slug = "".join(c for c in slug if c.isalnum() or c in "-_")
        dir_name = f"{self.id}_{slug}"
        research_dir = Path(data_dir) / dir_name
        research_dir.mkdir(parents=True, exist_ok=True)

        # Report
        if self.final_report:
            (research_dir / "report.md").write_text(
                self.final_report, encoding="utf-8"
            )

        # Sources
        sources_dir = research_dir / "sources"
        sources_dir.mkdir(exist_ok=True)
        for i, source in enumerate(self.sources, 1):
            safe_title = "".join(
                c for c in source.title[:60] if c.isalnum() or c in "-_ "
            ).strip().replace(" ", "_")
            filename = f"{i:03d}_{safe_title}.md"
            content = (
                f"# {source.title}\n"
                f"URL: {source.url}\n"
                f"Type: {source.source_type.value}\n"
                f"---\n\n"
                f"{source.content[:10000]}"
            )
            (sources_dir / filename).write_text(content, encoding="utf-8")

        # Extracts
        extracts_dir = research_dir / "extracts"
        extracts_dir.mkdir(exist_ok=True)
        # One file for all rounds: extracts do not record their round.
        extract_text = self._format_extracts() if self.harvest_results else ""
        if extract_text:
            (extracts_dir / "extracts.md").write_text(extract_text, encoding="utf-8")

        # Meta-Info
        meta = {
            "id": self.id,
            "title": self._generate_title(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 1),
            "n_sources": len(self.sources),
            "n_extracts": len(self.extracts),
            "rounds": self.rounds_completed,
            "query_preview": self.query[:200],
            "status": self.status,
            "search_strategy": self.search_stats,
        }
        (research_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Full context (for resuming)
        self._save_full_context(research_dir / "run.json")

        return str(research_dir)

    def _generate_title(self) -> str:
        if self.output_schema and self.output_schema.title:
            return self.output_schema.title
        return self.query[:80]

    def _format_extracts(self) -> str:
        """All relevant extracts of the run, grouped by source."""
        parts = []
        for hr in self.harvest_results:
            if not hr.is_relevant:
                continue
            parts.append(f"## {hr.source_title}\n**URL:** {hr.source_url}\n")
            for ext in hr.extracts:
                parts.append(f"- [{ext.question_id}] {ext.fact}")
                if ext.context:
                    parts.append(f"  Context: {ext.context}")
            parts.append("")
        return "\n".join(parts)

    def _save_full_context(self, path: Path):
        """Serialise the context as JSON (simplified).

        Also stores `research_plan` and `output_schema` via `to_dict()`,
        so that stored research runs can be replayed (e.g. for revisiting
        plans or analysing edge-case plans).
        """
        data = {
            "id": self.id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "rounds_completed": self.rounds_completed,
            "sources_count": len(self.sources),
            "extracts_count": len(self.extracts),
            "final_report_length": len(self.final_report),
            "search_stats": self.search_stats,
            "research_plan": (
                self.research_plan.to_dict()
                if self.research_plan else None
            ),
            "output_schema": (
                self.output_schema.to_dict()
                if self.output_schema else None
            ),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")


# ─── Output templates ────────────────────────────────────────────────

OUTPUT_TEMPLATES = {
    "general": {
        "label": "🔎 Allgemeine Recherche",
        "format": "free",
        "description": "The format is chosen automatically from the request — "
                        "the default mode for open research questions",
        "per_section": [],
        "synthesis_guidance": (
            "Write a readable report in connected paragraphs. "
            "Start with a short framing of the topic (2–3 sentences: what is it "
            "about, why does it matter?). Then the core findings ordered by relevance "
            "— each aspect as its own paragraph with a topic sentence, evidence "
            "and interpretation. Connect the paragraphs with transitions of content. "
            "End with a short conclusion (3–4 sentences) that brings the most important findings "
            "together. Use lists ONLY for concrete enumerations (software "
            "versions, recommended actions), not for individual facts."
        ),
    },
    "summary": {
        "label": "📝 Zusammenfassung",
        "format": "summary",
        "description": "Compact summary with the key statements, "
                        "sources embedded as inline links in the text",
        "per_section": ["Key statement", "Details"],
        "synthesis_guidance": (
            "Write a concise summary of at most 1–2 pages. "
            "Start with the central finding in 2–3 sentences (the most important "
            "first). Then the findings as connected paragraphs — each paragraph has "
            "one topic and leads on to the next. Avoid redundancy: mention each "
            "piece of information only once. Close with a short outlook "
            "or the most important open questions. No lists — everything "
            "as prose."
        ),
    },
    "structured_overview": {
        "label": "📋 Strukturierte Übersicht",
        "format": "structured_report",
        "description": "Per topic block: framing, research findings "
                        "and open questions — well structured and readable",
        "per_section": ["Framing", "Research findings",
                        "Open questions"],
        "synthesis_guidance": (
            "Write a clearly structured report with one section per "
            "topic block. Each section starts with a short framing "
            "(What is the topic? Why does it matter?) as a prose paragraph. Then "
            "the research findings as connected paragraphs — "
            "not as loose bullet points, but connected by argument. "
            "Open questions or recommended actions may appear as a short "
            "numbered list, but only at the end of each section. "
            "A short transition between the sections."
        ),
    },
    "comparison": {
        "label": "📊 Vergleichende Analyse",
        "format": "comparison",
        "description": "Systematic comparison with an overview table "
                        "and explanatory paragraphs per criterion",
        "per_section": ["Overview", "Detailed comparison", "Bewertung"],
        "synthesis_guidance": (
            "Start with a Markdown table as an overview (criteria as "
            "rows, options as columns). Then write an explanatory paragraph "
            "for EVERY criterion: what does the difference mean in concrete terms? "
            "Which source supports it? The table alone is not enough — the "
            "paragraphs provide the context. Close with an overall assessment: "
            "which option suits which purpose?"
        ),
    },
    "fact_check": {
        "label": "🔍 Faktencheck",
        "format": "factcheck",
        "description": "Claim → evidence → assessment with a clear verdict",
        "per_section": ["Claim", "Evidence", "Bewertung"],
        "synthesis_guidance": (
            "One section per claim. First state the claim to be checked "
            "clearly and unambiguously. Then the evidence as "
            "prose with embedded source links — what do the sources say, "
            "where do they agree, where do they contradict each other? At the end of each "
            "section a clear verdict marked with an emoji: "
            "✅ Confirmed / ⚠️ Partly true / ❌ Refuted / "
            "❓ Not clearly supported — each with a short justification."
        ),
    },
    "technical_doc": {
        "label": "📄 Technische Dokumentation",
        "format": "technical_doc",
        "description": "Technical analysis with code references, "
                        "configuration examples and recommendations",
        "per_section": ["Background", "Technical details",
                        "Configuration", "Empfehlung"],
        "synthesis_guidance": (
            "Write as for a technical wiki or internal "
            "documentation. Each section starts with a context paragraph "
            "(Why is this relevant? What is the goal?). Technical details "
            "as prose with code blocks (```), concrete version numbers, "
            "paths and configuration parameters. Always explanatory text between "
            "code blocks. Recommendations at the end as a numbered list "
            "with concrete steps. Avoid vague statements — "
            "every recommendation must be actionable."
        ),
    },
}


#: Template used when none (or an unknown one) is selected.
DEFAULT_TEMPLATE = "general"


def template_choices() -> list[tuple[str, str]]:
    """(label, id) pairs for the template dropdown."""
    return [(t["label"], tid) for tid, t in OUTPUT_TEMPLATES.items()]

