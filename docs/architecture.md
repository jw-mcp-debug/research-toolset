# Architecture

research-toolset is a single Python process: a Gradio interface
(`src/ui/gradio_app.py`) on top of a research orchestrator
(`src/pipeline/orchestrator.py`) and an analysis pipeline
(`src/pipeline/analysis_pipeline.py`). External services are reached through
connectors (`src/connectors/`) and two LLM clients (`src/llm/client.py`).

```
app.py ─ Gradio UI ─┬─ ResearchOrchestrator ── research pipeline (web / institution)
                    │        └─ AnalysisPipelineRunner ── analysis modes
                    └─ LiteratureChecker ─────────────── bibliography check
                             │
            connectors: SearXNG, web scraper, GitHub, GitLab, Elasticsearch,
            Solr, local files, literature APIs, person directory
            LLMs: primary (planning, synthesis) + harvest (extraction, classifiers)
```

## Two models

The `DualLLMClient` holds a *primary* model for the expensive, reasoning-heavy
steps (planning, synthesis, revision) and a *harvest* model for the many small
calls (fact extraction, classifiers). Both are OpenAI-compatible endpoints; the
harvest model defaults to the primary one. A semaphore
(`HARVEST_MAX_PARALLEL`) limits concurrent harvest calls.

Reasoning models are handled explicitly: reasoning tokens are read from the
separate reasoning field, a response without visible text raises
`EmptyLLMResponseError` instead of passing an empty string on, and the token
budget is escalated once when reasoning used it up.

## Research pipeline

The research modes run as a sequence of pipeline nodes (`src/pipeline/dag.py`,
`src/pipeline/dag_nodes.py`). Each node reads from and writes to one shared
`HarvestContext` (`src/pipeline/models.py`); every node result is recorded in
`ctx.node_results` and shown in the Pipeline run tab.

1. **Plan** — `FormatAgentNode` chooses the report format and sections,
   `AnalysisNode` turns the request into research questions with search terms
   per language, direct URLs, repositories and person-directory queries. The
   plan is validated deterministically (`plan_validation.py`). If the user
   ticked *Confirm plan*, the run stops here and resumes after confirmation.
2. **Research rounds** — per round:
   `SearchAndFetchNode` searches (SearXNG, optionally GitHub, GitLab,
   Elasticsearch, Solr, person directory, OpenAlex), ranks candidates
   (optionally with a reranker), fetches the best ones, removes duplicates and
   listing pages and follows links from primary sources;
   `QueryAnchorNode` decides whether the request is about a specific person;
   `HarvestNode` extracts facts per research question and source;
   `CoverageNode` assesses per question whether it is answered;
   `ContinueDecisionNode` decides whether another round is worthwhile.
   If it is, the next round searches for what the coverage assessment left
   open (`src/pipeline/followup.py`): each missing aspect of a partially
   answered question becomes a query, anchored with the question's own
   search term per language; an unanswered question is searched again with
   its text. Answered and filter-blocked questions are not searched again.
   The loop ends at `MAX_RESEARCH_ROUNDS`, when the classifier stops it, when
   nothing is left to search for, or when a round finds no new sources.
3. **Report** — `ContradictionCheckNode` looks for contradicting extracts,
   `DiagnosisNode` explains how the run went, `SynthesisNode` writes the report
   (map phase: one condensed answer per question; reduce phase: the report),
   `DiagnosisBannerNode` adds the diagnosis if it is problematic.
4. **Checks** — `FactoidVerificationNode` extracts concrete statements from the
   report and checks each against the extracts; `ReportRevisionNode` corrects
   statements contradicted with high confidence; `ReportQualityNode` and
   `QueryFulfillmentNode` assess the report as a whole;
   `ReportWarningBannerNode` writes their findings into the report.

### Classifiers instead of heuristics

Decisions that depend on meaning are made by small LLM classifiers
(`src/pipeline/classifiers/`), not by keyword rules. Every classifier returns a
typed result with a confidence, logs its call, and falls back to a
conservative default when the call fails or the confidence is low — for
example: no person filter, keep the source, "partially answered", "continue".
A heuristic on capitalised words, for instance, would read the German request
"DGX B300 vs B200 Preis-Leistungs-Verhältnis" as a person search and discard
every extract.

### Filters

Filters (`src/pipeline/filters/`) share one protocol: each decides itself
whether it applies, and each reports statistics. The person filter, for
example, only runs when the query-anchor classifier recognised a person with
confidence ≥ 0.7, and then keeps only sources that mention the person's
name. Filter losses are visible in the interface; a filter that discards
almost everything shows up in the diagnosis as `filter_too_strict`.

### Extract polarity

The harvest prompt marks every extract as `[POSITIVE]`, `[NEGATIVE]` (the
source explicitly does not cover the point) or `[META]` (a statement about the
source itself). Only positive extracts go into the synthesis; negative and
meta extracts stay in the harvest results and
are counted in the diagnosis.

## Analysis pipeline

The analysis modes (`src/pipeline/analysis_pipeline.py`) follow a different
pattern: a `Decomposer` turns the validated form inputs into a `TaskPlan` of
sub-tasks with dependencies; `TaskPlan.topological_order()` yields layers of
independent tasks; one `AnalysisLayerNode` per layer runs its tasks in
parallel, injecting the outputs of the tasks they depend on. Sub-tasks are
normally LLM prompts; two executors run without a model:
`literature_search` queries OpenAlex, Semantic Scholar and arXiv, and
`render_selection` joins a model's selection with the bibliographic data
returned by the APIs, so that authors, journals and DOIs are never copied by
the model. The final report is assembled from the tasks nothing else depends
on (plus tasks marked `include_in_report`), with a note on failed tasks.

## Bibliography check

`src/pipeline/literature_check.py` splits a pasted bibliography into entries
(several segmentation strategies, the one with the most plausible result
wins), lets the model parse each entry into fields, looks every entry up in
the literature APIs in parallel, compares fields, verifies URLs of entries not
found, and builds the report deterministically; only short per-entry comments
come from the model.

## Output language

Text the tool itself writes into reports and exports comes from catalogs
(`src/locales/*.toml`, see [output languages](output-languages.md)). Prompts
are English; every prompt whose output reaches the report ends with an
instruction naming the output language. The language of a run is set once and
held in a context variable, so parallel sessions do not interfere.

## Stopping

One `StopSignal` object per run is passed to every sub-pipeline and checked
before every node and between analysis layers; the stop button sets it.

## Security-relevant parts

- `src/core/url_security.py` — SSRF protection for every fetched URL, also
  after DNS resolution and on redirects.
- `src/core/tls.py` — outgoing TLS verification (on by default).
- `src/core/retention.py` — deletes stored runs after `CLEANUP_MAX_AGE_DAYS`.

## Persistence

Each completed run is written to `DATA_DIR/<id>_<slug>/`: `report.md`,
`sources/`, `extracts/`, `meta.json` and `run.json` (the full context). The
tool writes these files for inspection and export; it does not read them back.
