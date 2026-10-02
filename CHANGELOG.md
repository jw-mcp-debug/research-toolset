# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased] — fork jw-mcp-debug/research-toolset

Changes on the `de-ui` branch of this fork, on top of upstream 1.0.0.

### Added

- **Chat persistence**: the chat — what is shown and the context sent to the
  LLM — is kept in the browser's localStorage (encrypted, `gr.BrowserState`)
  and restored after a page reload or a closed tab. It is written once at the
  end of each action, not per streamed chunk; greeting and placeholders are
  not stored, and the size is capped.
- **Result persistence**: the last finished result (report, sources,
  progress, extracts, pipeline run, export version of the report, and the
  BibTeX of a reference check) is kept the same way, up to 250 KB (the
  pipeline view is shortened first, the report last). A restored result
  opens the result area and is marked as restored; Markdown, Word (without
  the metadata appendices) and BibTeX export work again.
- `CHAT_STORAGE_SECRET`: fixed key for the browser storage. Without it Gradio
  picks a random key per start, and every restart makes the stored chat and
  result unreadable.
- **Interface languages**: English and German, switched in the header. Each
  language is its own page (default at the root, others under `/<code>`);
  the browser remembers the choice. Interface text stays English in the code
  (`tr("…")`), translations are catalogs in `src/ui/locales/` with the
  English text as key. A test fails for untranslated text and mismatched
  placeholders. New variables `UI_LANGUAGES` and `DEFAULT_UI_LANGUAGE`.

### Changed

- **Calmer layout**: text field and toolbar form one input card, *Start
  research* is the only accent button, the duplicate send arrow is gone.
- Report template, report language and the check boxes sit in a collapsible
  options row; only the options that apply to the selected mode are shown.
- Export moved into the header of the result area; the download field
  appears only once there is a file. The tabs Progress, Extracts and
  Pipeline run are combined into *History*, next to *Report* and *Sources*.
- Header buttons are labelled; *New chat* lives only in the header. Buttons
  14 px text / 36 px high, one blue accent, consistent radii.
- *New chat* discards the stored chat and result and clears the result
  area; a research run still in progress keeps its result area.

### Fixed

- BibTeX export of a reference check never worked: it read
  `search_stats["entries"]`, which nothing wrote. The check's own BibTeX
  (best API match per entry) is now passed on as `search_stats["bibtex"]`.

## [1.0.0] — 2026-09-28

First public release.

### Research

- Web research: plan with research questions and multilingual search terms,
  search via SearXNG (optionally GitHub, GitLab, Elasticsearch, OpenAlex),
  ranking, fetching, link following, fact extraction per question, and a report
  with inline source links.
- Follow-up rounds search for the aspects the coverage assessment reports as
  missing (`MAX_RESEARCH_ROUNDS`).
- Optional plan confirmation before the research starts.
- Institution mode (optional): institution profile, person directory with a
  consent model, Solr or Elasticsearch index of the institution's website.

### Analysis modes

- In-depth explanation, peer review, decision analysis, research design, grant
  proposal (draft) and literature review, run as dependency graphs of sub-tasks;
  the literature-based modes query OpenAlex, Semantic Scholar and arXiv.
- Find literature: assessed selection with bibliographic details taken from the
  APIs.

### Bibliography check

- Entry-by-entry check against CrossRef, OpenAlex, Semantic Scholar, arXiv,
  DBLP and OpenLibrary, URL verification, duplicate detection, corrected
  bibliography in APA and DIN 1505-2, BibTeX export.

### Quality safeguards

- LLM classifiers for query anchor, source relevance, coverage, continue
  decision, search scope and diagnosis, each with a conservative fallback.
- Factoid verification against the collected extracts, revision of
  contradicted statements, report quality and fulfilment checks written into
  the report; pipeline-run view with every decision.

### Output languages

- English interface; reports and exports in English, German or any language
  added as a catalog file (`OUTPUT_LANGUAGES`, `OUTPUT_CATALOG_DIR`).

### Operation

- Word and Markdown export, BibTeX export.
- Privacy defaults: no telemetry, no external fonts, no share links, no public
  API; retention of stored runs (`CLEANUP_MAX_AGE_DAYS`).
- SSRF protection for every fetched URL, TLS verification on by default,
  binding to `127.0.0.1` by default, Docker image running as an unprivileged user.

[Unreleased]: https://github.com/jw-mcp-debug/research-toolset/compare/v1.0.0...de-ui
[1.0.0]: https://github.com/maltedreyer-5/research-toolset/releases/tag/v1.0.0
