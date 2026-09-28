"""Non-LLM executor: a real literature search for analysis tasks.

Why this exists: `AnalysisLayerNode` can only send prompts to a language
model. Without this executor, the "find literature" mode would produce
search terms and Boolean queries and stop there — no database would be
queried and nothing evaluated.

This module actually runs the queries, via the existing
`LiteratureSearchService` (OpenAlex + Semantic Scholar + arXiv in
parallel, with de-duplication).

Split: only logic that is testable without network lives here — query
extraction, result formatting, error handling. The actual HTTP traffic
is in the service, which is injected.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

import contextvars

from src.output_language import t as _catalog_t

# Output language of the report being produced (set by the analysis layer
# node before it runs a search or render task).
OUTPUT_LANG: contextvars.ContextVar[str] = contextvars.ContextVar("search_output_lang", default="en")


def _t(key: str, **values) -> str:
    return _catalog_t(key, OUTPUT_LANG.get(), **values)

# Upper bounds, so that a misguided plan does not trigger hundreds
# of API requests.
MAX_QUERIES = 8
MAX_RESULTS_PER_QUERY = 25
MAX_TOTAL_RESULTS = 60


def extract_queries(
    raw_output: str, parsed_output: Optional[dict] = None,
) -> list[str]:
    """Get the search queries from the result of the preceding task.

    Prefers the structured form (`output_format="json"` yields
    `parsed_output`) because it is unambiguous. Falls back to line
    parsing so that a plain-text model result is not lost entirely.

    Deliberately conservative: better a few good queries than many
    fragments of running text.
    """
    if parsed_output:
        for key in ("queries", "search_queries", "anfragen"):
            value = parsed_output.get(key)
            if isinstance(value, list) and value:
                out = [str(q).strip() for q in value if str(q).strip()]
                return out[:MAX_QUERIES]

    text = (raw_output or "").strip()

    # If the raw text looks like JSON, the line fallback is pointless and
    # even harmful: from an object with unexpected keys, something like
    # `"nonsense": 1` would be sent to the literature APIs as a query.
    # Better no query from this source at all — then the caller's
    # fallback query applies.
    if text.startswith("{") or text.startswith("["):
        return []

    queries: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip bullets and numbering
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
        # Remove Markdown bold and code ticks
        line = line.replace("**", "").strip("`").strip()
        if not line or len(line) < 4:
            continue
        # Weed out headings and running text: a search query does not end
        # with a colon and is not very long.
        if line.endswith(":") or len(line) > 200:
            continue
        queries.append(line)
    return queries[:MAX_QUERIES]


def format_results_markdown(results: list) -> str:
    """Format hits as a readable, referenced list.

    Deliberately includes DOI and year: the list is the report's
    evidence and must be checkable without a model copying it (and
    inventing identifiers while doing so).
    """
    if not results:
        return _t("search.no_hits")

    lines = [_t("search.n_hits_one" if len(results) == 1 else "search.n_hits", n=len(results)), ""]
    for i, r in enumerate(results, 1):
        title = (getattr(r, "title", "") or _t("search.untitled")).strip()
        authors = getattr(r, "authors", None) or []
        year = getattr(r, "year", 0) or 0
        venue = (getattr(r, "venue", "") or "").strip()
        doi = (getattr(r, "doi", "") or "").strip()
        url = (getattr(r, "url", "") or "").strip()
        cites = getattr(r, "cited_by_count", 0) or 0
        sources = getattr(r, "sources", None) or []

        who = ", ".join(str(a) for a in authors[:3])
        if len(authors) > 3:
            who += " et al."

        head = f"{i}. **{title}**"
        if year:
            head += f" ({year})"
        lines.append(head)

        meta = []
        if who:
            meta.append(who)
        if venue:
            meta.append(venue)
        if cites:
            meta.append(_t("search.citations", n=cites))
        if sources:
            meta.append("via " + ", ".join(str(s) for s in sources))
        if meta:
            lines.append("   " + " · ".join(meta))

        if doi:
            lines.append(f"   DOI: {doi}")
        elif url:
            lines.append(f"   {url}")
        lines.append("")
    return "\n".join(lines).rstrip()


def results_to_dicts(results: list) -> list[dict]:
    """Structured version for `TaskResult.parsed_output`."""
    out = []
    for r in results:
        out.append({
            "title": getattr(r, "title", ""),
            "authors": list(getattr(r, "authors", None) or []),
            "year": getattr(r, "year", 0),
            "venue": getattr(r, "venue", ""),
            "doi": getattr(r, "doi", ""),
            "url": getattr(r, "url", ""),
            "cited_by_count": getattr(r, "cited_by_count", 0),
            "sources": list(getattr(r, "sources", None) or []),
            "abstract": (getattr(r, "abstract", "") or "")[:1000],
        })
    return out


def render_selected_references(selection: dict, results: list[dict]) -> str:
    """Render the selected works with full details.

    The model selects and justifies; the bibliographic data comes from
    this function — i.e. from the API response, not from the model. A
    selection thus cannot contain invented authors, journals or DOIs,
    and the details are complete instead of being cut down to title
    and year.

    A selection whose DOI does not occur in the hits is discarded and
    counted at the end — that is the hallucination detector.

    Args:
        selection: parsed model answer with `selected`, `gaps`,
            `next_round`.
        results: the structured hits from the search task.
    """
    by_doi, by_title = {}, {}
    for r in results or []:
        doi = (r.get("doi") or "").strip().lower()
        if doi:
            by_doi[doi] = r
        title = (r.get("title") or "").strip().lower()
        if title:
            by_title[title] = r

    picks = selection.get("selected") or selection.get("auswahl") or []
    lines: list[str] = []
    dropped = 0

    for i, pick in enumerate(picks, 1):
        if not isinstance(pick, dict):
            continue
        doi = str(pick.get("doi") or "").strip().lower()
        title = str(pick.get("title") or pick.get("titel") or "").strip()
        hit = by_doi.get(doi) or by_title.get(title.lower())
        if hit is None:
            dropped += 1
            continue

        authors = hit.get("authors") or []
        who = ", ".join(str(a) for a in authors[:5])
        if len(authors) > 5:
            who += " et al."

        lines.append(f"**{i}. {hit.get('title') or 'ohne Titel'}**")
        meta: list[str] = []
        if who:
            meta.append(who)
        if hit.get("year"):
            meta.append(str(hit["year"]))
        if hit.get("venue"):
            meta.append(f"*{hit['venue']}*")
        if meta:
            lines.append("   " + " · ".join(meta))

        detail = []
        if hit.get("cited_by_count"):
            detail.append(_t("search.citations", n=hit['cited_by_count']))
        if hit.get("sources"):
            detail.append(_t("search.found_via") + ", ".join(
                str(s) for s in hit["sources"]))
        if detail:
            lines.append("   " + " · ".join(detail))
        if hit.get("doi"):
            lines.append(f"   DOI: [{hit['doi']}]"
                         f"(https://doi.org/{hit['doi']})")
        elif hit.get("url"):
            lines.append(f"   {hit['url']}")

        reason = str(pick.get("reason") or pick.get("begruendung") or "").strip()
        if reason:
            lines.append(f"   → {reason}")
        lines.append("")

    parts: list[str] = []
    if lines:
        parts.append(_t("search.most_relevant"))
        parts.extend(lines)
    else:
        parts.append(_t("search.no_selection"))

    gaps = str(selection.get("gaps") or selection.get("luecken") or "").strip()
    if gaps:
        parts.append(_t("search.gaps"))
        parts.append(gaps)
        parts.append("")

    nxt = str(selection.get("next_round")
              or selection.get("empfehlung") or "").strip()
    if nxt:
        parts.append(_t("search.next_round"))
        parts.append(nxt)
        parts.append("")

    if dropped:
        parts.append(
            _t("search.dropped_one" if dropped == 1 else "search.dropped_many", n=dropped)
        )
    return "\n".join(parts).rstrip()


def _seed_dois(results: list, max_seeds: int) -> list[str]:
    """Pick the seeds for citation chasing.

    The criterion is the citation count: highly cited works are the
    better starting points because their citation networks are denser.
    Without a DOI a hit is useless as a seed.
    """
    with_doi = [r for r in results if (getattr(r, "doi", "") or "").strip()]
    with_doi.sort(key=lambda r: getattr(r, "cited_by_count", 0) or 0,
                  reverse=True)
    seen, seeds = set(), []
    for r in with_doi:
        doi = r.doi.strip()
        if doi in seen:
            continue
        seen.add(doi)
        seeds.append(doi)
        if len(seeds) >= max_seeds:
            break
    return seeds


async def run_literature_search(
    queries: list[str],
    search_service: Any,
    config: Optional[dict] = None,
) -> tuple[str, dict]:
    """Run the search queries and return (markdown, structure).

    A failure of a single query does not end the whole search —
    literature APIs regularly fail individually, and a partial result
    is worth much more than an error message.
    """
    config = config or {}
    if not queries:
        return _t("search.no_queries"), {"results": [], "queries": []}
    if search_service is None:
        return (
            _t("search.unavailable"),
            {"results": [], "queries": queries, "error": "no_service"},
        )

    limit = min(int(config.get("limit", 20) or 20), MAX_RESULTS_PER_QUERY)
    kwargs = {"limit": limit}
    for key in ("year_from", "year_to", "min_citations",
                "peer_reviewed_only", "sources"):
        if config.get(key) not in (None, "", []):
            kwargs[key] = config[key]

    collected: list = []
    errors: list[str] = []
    used: list[str] = []

    for query in queries[:MAX_QUERIES]:
        try:
            hits = await search_service.multi_source_query(query, **kwargs)
            collected.extend(hits or [])
            used.append(query)
        except Exception as e:
            logger.warning("Literature search failed for %r: %s",
                           query, type(e).__name__)
            errors.append(f"{query}: {type(e).__name__}")

    # ── Citation chasing ──
    # The most-cited hits of the query search serve as seeds for forward
    # and backward citations. That finds works none of the formulated
    # queries would have caught — in a systematic review it is the
    # difference between a sample and coverage.
    #
    # Runs serially and takes noticeable time (~30-90 s according to the
    # service), hence only on explicit request and with a limited number
    # of seeds.
    n_expanded = 0
    if config.get("expand_citations") and collected:
        seeds = _seed_dois(collected, int(config.get("max_seeds", 5) or 5))
        if seeds:
            try:
                expansion = await search_service.expand_from_seeds(
                    seeds,
                    max_per_seed=int(config.get("max_per_seed", 15) or 15),
                )
                n_expanded = len(expansion or [])
                collected.extend(expansion or [])
                logger.info("Citation chasing: %d seeds → %d further hits",
                            len(seeds), n_expanded)
            except Exception as e:
                logger.warning("Citation chasing failed: %s",
                               type(e).__name__)
                errors.append(f"citation_chasing: {type(e).__name__}")

    try:
        from src.connectors.literature_search import dedupe_results
        collected = dedupe_results(collected)
    except Exception as e:
        logger.warning("Dedupe failed: %s", type(e).__name__)

    collected.sort(
        key=lambda r: getattr(r, "cited_by_count", 0) or 0, reverse=True,
    )
    collected = collected[:MAX_TOTAL_RESULTS]

    markdown = format_results_markdown(collected)
    if used:
        # The queries actually run belong visibly in the report: they are
        # traceable and reusable — the result of this mode is a search
        # protocol as much as a list of papers.
        head = _t("search.queries_run") + "; ".join(f"`{q}`" for q in used)
        if n_expanded:
            head += (_t("search.plus_citation_chasing", n=n_expanded))
        markdown = head + "_\n\n" + markdown
    if errors:
        markdown += (
            _t("search.some_failed", failed=len(errors), total=len(queries))
        )
    if not collected and errors:
        markdown = (
            _t("search.all_failed")
        )

    return markdown, {
        "results": results_to_dicts(collected),
        "queries": used,
        "errors": errors,
        "n_results": len(collected),
        "n_expanded": n_expanded,
    }
