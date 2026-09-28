"""
Literature search service: query-based search and citation expansion.

Complements the LiteratureAPIClient (which primarily verifies
bibliographies) with a high-level service for:

1. multi-source query search (OpenAlex + Semantic Scholar + arXiv)
2. citation chasing (forward + backward from seed papers)
3. de-duplication with provenance tracking (preparing PRISMA)
4. ranking by citation count

Design decisions:
- primary source: OpenAlex (best coverage, best citation API)
- secondary source: Semantic Scholar (best abstracts)
- tertiary source: arXiv (only for CS/physics-related queries)
- de-duplication by DOI, fallback to normalised title similarity
- provenance as a LIST per paper: a paper can be discovered several
  times (e.g. by query as well as by citation expansion)
- rate limiting: citation expansion runs SERIALLY, not in parallel, to
  stay reliably within the APIs' limits
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.connectors.literature_apis import LiteratureAPIClient

logger = logging.getLogger(__name__)


# ─── Data model ────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A uniform search hit from any literature API.

    Difference from LiteratureEntry: this type is meant for discovery,
    not verification. It contains:
    - all metadata needed for relevance decisions
    - an abstract (delivered by the APIs, no separate fetch)
    - a LIST of provenances (query, citation forward, citation backward)
    - metadata preparing PRISMA (seed_distance)
    """

    # ─── Identity ───
    # Primary identifier (DOI preferred, otherwise an external ID from the source)
    id: str = ""
    doi: str = ""

    # ─── Bibliographic metadata ───
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int = 0
    venue: str = ""              # journal, conference, preprint server
    abstract: str = ""
    url: str = ""

    # ─── Relevance signals ───
    cited_by_count: int = 0
    is_peer_reviewed: Optional[bool] = None  # None = unknown
    paper_type: str = ""         # article, book, preprint, etc.

    # ─── API sources ───
    # A list, because the same paper can be found in several APIs
    sources: list[str] = field(default_factory=list)  # ["openalex", "s2", "arxiv"]

    # ─── Discovery tracking (preparing PRISMA) ───
    # List of all provenances under which this paper was found.
    # Format:
    #   "query:agile teams"
    #   "forward_from:10.1000/xxxx"
    #   "backward_from:10.1000/yyyy"
    discovered_via: list[str] = field(default_factory=list)

    # Minimum distance to a user seed (0 = direct query hit,
    # 1 = found via a seed citation, 2 = transitive expansion ...)
    seed_distance: int = 0

    # ─── Derived properties ───

    @property
    def discovery_count(self) -> int:
        """How often was this paper discovered independently?

        Only a metadatum for the PRISMA log / debugging. NOT used as a
        ranking signal (design decision: aggressive de-duplication
        without a ranking boost).
        """
        return len(self.discovered_via)

    @property
    def relevance_score(self) -> float:
        """Ranking signal for sorting.

        Citation count only. Multiple discovery gives NO boost (design
        decision: aggressive de-duplication).
        """
        return float(self.cited_by_count)

    def merge_from(self, other: "SearchResult") -> None:
        """Merge data from a duplicate hit into this entry.

        Used during de-duplication when the same paper comes from several
        sources or provenances. Rules:
        - metadata: keep the more complete one (longer strings)
        - lists (authors, sources, discovered_via): union, de-duplicated
        - cited_by_count: maximum
        - seed_distance: minimum (shortest path)
        """
        # Fill missing fields
        if not self.doi and other.doi:
            self.doi = other.doi
        if len(other.title) > len(self.title):
            self.title = other.title
        if len(other.abstract) > len(self.abstract):
            self.abstract = other.abstract
        if not self.venue and other.venue:
            self.venue = other.venue
        if not self.url and other.url:
            self.url = other.url
        if not self.year and other.year:
            self.year = other.year
        if len(other.authors) > len(self.authors):
            self.authors = other.authors

        # Citations: the maximum (APIs may differ)
        self.cited_by_count = max(self.cited_by_count, other.cited_by_count)

        # Peer-review status: if one source says "True", trust it
        if other.is_peer_reviewed is True:
            self.is_peer_reviewed = True
        elif self.is_peer_reviewed is None:
            self.is_peer_reviewed = other.is_peer_reviewed

        # Merge lists (keep the order, drop duplicates)
        for src in other.sources:
            if src not in self.sources:
                self.sources.append(src)
        for origin in other.discovered_via:
            if origin not in self.discovered_via:
                self.discovered_via.append(origin)

        # Shortest path to a seed
        self.seed_distance = min(self.seed_distance, other.seed_distance)


# ─── De-duplication helpers ──────────────────────────────────────────

def _normalize_title(title: str) -> str:
    """Normalise a title for similarity comparisons.

    Removes punctuation, normalises whitespace, lower-cases.
    """
    import re
    normalized = title.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _titles_match(title_a: str, title_b: str, threshold: float = 0.90) -> bool:
    """Check whether two titles count as duplicates (similarity >= threshold)."""
    a = _normalize_title(title_a)
    b = _normalize_title(title_b)
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    """De-duplicate a list of SearchResults.

    Strategy:
    1. first pass: de-duplication by DOI (exact match)
    2. second pass: de-duplication by title similarity (>= 0.90) for
       entries without a DOI or with a differently written DOI

    For duplicate hits the data is merged via merge_from(), so that
    provenances and sources are combined as lists.
    """
    if not results:
        return []

    # ── Pass 1: DOI de-duplication ──
    by_doi: dict[str, SearchResult] = {}
    without_doi: list[SearchResult] = []

    for result in results:
        doi_key = result.doi.lower().strip() if result.doi else ""
        if doi_key:
            if doi_key in by_doi:
                by_doi[doi_key].merge_from(result)
            else:
                by_doi[doi_key] = result
        else:
            without_doi.append(result)

    unique_list = list(by_doi.values())

    # ── Pass 2: title similarity for entries without a DOI ──
    # (also against the DOI clusters, in case a duplicate exists in both forms)
    for result in without_doi:
        matched = False
        for existing in unique_list:
            if _titles_match(result.title, existing.title):
                existing.merge_from(result)
                matched = True
                break
        if not matched:
            unique_list.append(result)

    return unique_list


# ─── Orchestrator-Service ───────────────────────────────────────────

class LiteratureSearchService:
    """High-level service for literature discovery and citation expansion.

    Uses the LiteratureAPIClient for the low-level HTTP calls and adds:
    - multi-source query search (in parallel over OpenAlex + S2 + arXiv)
    - citation chasing (forward + backward, serially to respect API limits)
    - a uniform SearchResult structure
    - de-duplication with merging of provenances

    Example:
        service = LiteratureSearchService(api_client)

        # Phase 1: query search
        query_hits = await service.multi_source_query(
            "agile teams hybrid work", limit=30,
            year_from=2018, min_citations=5,
        )

        # Phase 2: citation expansion from seeds
        seeds = ["10.1000/xxxx", "10.1000/yyyy"]
        expansion = await service.expand_from_seeds(
            seeds, max_per_seed=20,
        )

        # Combine
        all_results = dedupe_results(query_hits + expansion)
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
    """

    def __init__(self, api_client: "LiteratureAPIClient"):
        self.api = api_client

    async def multi_source_query(
        self,
        query: str,
        limit: int = 25,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0,
        peer_reviewed_only: bool = False,
        sources: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """Run a query-based search over several APIs in parallel.

        Args:
            query: natural-language search query (e.g. "hybrid work agile teams")
            limit: maximum hits PER SOURCE (the total may be larger until de-duplication)
            year_from: earliest year (inclusive)
            year_to: latest year (inclusive)
            min_citations: minimum citation count (post-filter)
            peer_reviewed_only: if True, filter on is_peer_reviewed=True
            sources: list of sources to use. Default: all three.
                     Possible values: "openalex", "semantic_scholar", "arxiv"

        Returns:
            De-duplicated list of SearchResults, sorted by relevance_score.
        """
        import asyncio

        active_sources = sources or ["openalex", "semantic_scholar", "arxiv"]
        tasks = []

        if "openalex" in active_sources:
            tasks.append(self.api.query_openalex(
                query, limit=limit,
                year_from=year_from, year_to=year_to,
            ))
        if "semantic_scholar" in active_sources:
            tasks.append(self.api.query_semantic_scholar(
                query, limit=limit,
                year_from=year_from, year_to=year_to,
            ))
        if "arxiv" in active_sources:
            tasks.append(self.api.query_arxiv(
                query, limit=limit,
                year_from=year_from, year_to=year_to,
            ))

        # Run in parallel (every API has its own rate limiter)
        results_per_source = await asyncio.gather(
            *tasks, return_exceptions=True,
        )

        # Flatten and sort out errors
        all_results: list[SearchResult] = []
        for result_set in results_per_source:
            if isinstance(result_set, Exception):
                logger.warning(f"Query source failed: {result_set}")
                continue
            all_results.extend(result_set)

        # Set the provenance marker
        origin = f"query:{query[:80]}"
        for r in all_results:
            if origin not in r.discovered_via:
                r.discovered_via.append(origin)

        # De-duplicate
        deduped = dedupe_results(all_results)

        # Post-Filter
        filtered = [
            r for r in deduped
            if r.cited_by_count >= min_citations
            and (not peer_reviewed_only or r.is_peer_reviewed is True)
        ]

        # Sort by relevance_score (citation count)
        filtered.sort(key=lambda r: r.relevance_score, reverse=True)

        return filtered

    async def expand_from_seeds(
        self,
        seed_dois: list[str],
        max_per_seed: int = 20,
        directions: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """Citation expansion from a list of seed DOIs.

        Runs SERIALLY (per seed and per direction one API call after the
        other), so as not to violate the literature APIs' rate limits.
        With 5 seeds and both directions that is ~10-15 API calls and a
        run time of ~30-90 seconds depending on the API rate.

        Args:
            seed_dois: list of DOIs that serve as starting points
            max_per_seed: maximum citations to fetch per seed per direction
            directions: ["forward", "backward"] (default: both)

        Returns:
            De-duplicated list of the papers found.
            Seed distance is 1 for direct citations.
        """
        if directions is None:
            directions = ["forward", "backward"]

        all_expansion: list[SearchResult] = []

        for seed_doi in seed_dois:
            seed_doi_clean = seed_doi.strip()
            if not seed_doi_clean:
                continue

            if "forward" in directions:
                try:
                    citers = await self.api.get_citers_openalex(
                        seed_doi_clean, limit=max_per_seed,
                    )
                    origin = f"forward_from:{seed_doi_clean}"
                    for c in citers:
                        if origin not in c.discovered_via:
                            c.discovered_via.append(origin)
                        c.seed_distance = 1
                    all_expansion.extend(citers)
                except Exception as e:
                    logger.warning(
                        f"Forward expansion for {seed_doi_clean} "
                        f"failed: {e}"
                    )

            if "backward" in directions:
                try:
                    refs = await self.api.get_references_openalex(
                        seed_doi_clean, limit=max_per_seed,
                    )
                    origin = f"backward_from:{seed_doi_clean}"
                    for r in refs:
                        if origin not in r.discovered_via:
                            r.discovered_via.append(origin)
                        r.seed_distance = 1
                    all_expansion.extend(refs)
                except Exception as e:
                    logger.warning(
                        f"Backward expansion for {seed_doi_clean} "
                        f"failed: {e}"
                    )

        return dedupe_results(all_expansion)

    async def search_with_expansion(
        self,
        query: Optional[str] = None,
        seed_dois: Optional[list[str]] = None,
        query_limit: int = 25,
        expansion_per_seed: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0,
        peer_reviewed_only: bool = False,
        max_total_results: int = 100,
    ) -> tuple[list[SearchResult], dict]:
        """Combined search: query + citation expansion.

        Uses the combination of inputs:
        - only query: plain query search
        - only seeds: plain expansion
        - both: query search PLUS expansion from seeds, de-duplicated

        Returns:
            (results, search_log)
            search_log contains metadata preparing PRISMA:
            - n_from_query, n_from_forward, n_from_backward
            - n_after_dedupe, n_after_filter
            - query_used, seeds_used
        """
        results: list[SearchResult] = []
        log = {
            "query_used": query or "",
            "seeds_used": seed_dois or [],
            "n_from_query": 0,
            "n_from_expansion": 0,
            "n_raw": 0,
            "n_after_dedupe": 0,
            "n_after_filter": 0,
        }

        if query:
            query_results = await self.multi_source_query(
                query, limit=query_limit,
                year_from=year_from, year_to=year_to,
                min_citations=0,  # filter only at the end
                peer_reviewed_only=False,  # filter only at the end
            )
            results.extend(query_results)
            log["n_from_query"] = len(query_results)

        if seed_dois:
            expansion = await self.expand_from_seeds(
                seed_dois, max_per_seed=expansion_per_seed,
            )
            results.extend(expansion)
            log["n_from_expansion"] = len(expansion)

        log["n_raw"] = len(results)

        # Overall de-duplication
        deduped = dedupe_results(results)
        log["n_after_dedupe"] = len(deduped)

        # Post-Filter
        filtered = [
            r for r in deduped
            if r.cited_by_count >= min_citations
            and (not peer_reviewed_only or r.is_peer_reviewed is True)
            and (not year_from or r.year >= year_from)
            and (not year_to or r.year <= year_to)
        ]
        log["n_after_filter"] = len(filtered)

        # Sort and limit
        filtered.sort(key=lambda r: r.relevance_score, reverse=True)
        filtered = filtered[:max_total_results]
        log["n_returned"] = len(filtered)

        return filtered, log
