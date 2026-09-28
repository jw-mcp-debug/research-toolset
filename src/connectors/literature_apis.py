"""
Academic literature APIs: arXiv, CrossRef, OpenAlex, Semantic Scholar, DBLP.

All APIs are free and usable without authentication. They are queried in
parallel and the results are merged. arXiv papers are looked up
preferentially via the arXiv API. Rate limiting is handled by the global
APIRateLimiter.
"""

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

import httpx

from src.output_language import tc
from src.connectors.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

# Polite-pool User-Agent (CrossRef asks for it and answers faster).
# Built per client from the institution profile / CONTACT_EMAIL.
from src.institution import polite_user_agent


@dataclass
class LiteratureEntry:
    """A parsed bibliography reference."""
    id: int = 0
    raw_text: str = ""
    # Parsed fields
    authors: list[str] = field(default_factory=list)
    title: str = ""
    year: str = ""
    journal: str = ""            # journal / conference / publisher
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    isbn: str = ""
    url: str = ""
    publisher: str = ""
    edition: str = ""
    entry_type: str = "article"  # article, book, inproceedings, thesis, web, other
    # check result
    status: str = "pending"      # pending, verified, deviations, not_found, error
    api_matches: list[dict] = field(default_factory=list)
    deviations: list[dict] = field(default_factory=list)
    corrected_fields: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    checked_sources: list[str] = field(default_factory=list)
    # Metrics
    cited_by_count: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "LiteratureEntry":
        """Read an entry from a dict (typically LLM JSON from the
        literature parser).

        Defensive: None values and wrong types are set to empty defaults.
        The author list is normalised. `id` is set afterwards by the
        caller (typically the index in the entry list, so that the
        numbering is continuous) — `from_dict` always returns `id=0`.
        """
        if not isinstance(d, dict):
            return cls()

        def _s(val, default=""):
            if val is None:
                return default
            return str(val).strip()

        authors = d.get("authors") or []
        if not isinstance(authors, list):
            authors = [str(authors)]

        return cls(
            raw_text=_s(d.get("raw_text")),
            authors=authors,
            title=_s(d.get("title")),
            year=_s(d.get("year")),
            journal=_s(d.get("journal")),
            volume=_s(d.get("volume")),
            issue=_s(d.get("issue")),
            pages=_s(d.get("pages")),
            doi=_s(d.get("doi")),
            isbn=_s(d.get("isbn")),
            url=_s(d.get("url")),
            publisher=_s(d.get("publisher")),
            edition=_s(d.get("edition")),
            entry_type=_s(d.get("entry_type"), "article"),
        )


@dataclass
class LiteratureReport:
    """Overall result of the literature check."""
    entries: list[LiteratureEntry] = field(default_factory=list)
    total: int = 0
    verified: int = 0
    with_deviations: int = 0
    not_found: int = 0
    errors: int = 0
    # Statistics
    api_stats: dict = field(default_factory=dict)
    llm_stats: dict = field(default_factory=dict)
    # BibTeX-Export (C1)
    bibtex: str = ""


class LiteratureAPIClient:
    """Client for academic literature APIs.

    Uses the global APIRateLimiter for rate-limited requests with
    automatic retry and exponential back-off.
    """

    def __init__(self, timeout: float = 15.0):
        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": polite_user_agent()},
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self.limiter = get_rate_limiter()

        # S2 API key (optional, raises the limit from 1/s to 10/s)
        self._s2_api_key = os.environ.get("S2_API_KEY", "")

    async def close(self):
        await self.client.aclose()

    async def _get(self, api_name: str, url: str, **kwargs) -> httpx.Response:
        """Rate-limited GET request via the global limiter.

        Adds the API key header automatically for S2.
        """
        headers = kwargs.pop("headers", {})
        if api_name == "s2" and self._s2_api_key:
            headers["x-api-key"] = self._s2_api_key
        if headers:
            kwargs["headers"] = headers
        return await self.limiter.request(
            api_name, self.client, "GET", url, **kwargs,
        )

    # ─── CrossRef ───────────────────────────────────────────────────

    async def search_crossref(self, entry: LiteratureEntry) -> dict | None:
        """Look up an entry in CrossRef (DOI registration)."""
        try:
            # With a known DOI: direct lookup
            if entry.doi:
                doi = self.validate_doi(entry.doi)
                if doi:
                    url = f"https://api.crossref.org/works/{doi}"
                resp = await self._get("crossref", url)
                if resp.status_code == 200:
                    data = resp.json()
                    item = data.get("message", {})
                    return self._parse_crossref_item(item)

            # Otherwise: bibliographic search
            params = {"rows": 3}
            if entry.title:
                params["query.bibliographic"] = entry.title
            if entry.authors:
                params["query.author"] = entry.authors[0]

            resp = await self._get("crossref",
                "https://api.crossref.org/works",
                params=params,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            items = data.get("message", {}).get("items", [])

            # Find the best match
            for item in items[:3]:
                parsed = self._parse_crossref_item(item)
                if parsed and self._is_likely_match(entry, parsed):
                    return parsed

            return None

        except Exception as e:
            logger.debug(f"CrossRef error: {e}")
            return None

    @staticmethod
    def _parse_crossref_item(item: dict) -> dict | None:
        """Parse a CrossRef item into normalised fields."""
        if not item:
            return None

        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""

        authors = []
        for a in item.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            if family:
                authors.append(f"{given} {family}".strip())

        # Extract the date
        date_parts = (
            item.get("published-print", {}).get("date-parts", [[]])
            or item.get("published-online", {}).get("date-parts", [[]])
            or item.get("issued", {}).get("date-parts", [[]])
        )
        year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""

        journal_list = item.get("container-title", [])
        journal = journal_list[0] if journal_list else ""

        return {
            "source": "CrossRef",
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journal,
            "volume": item.get("volume", ""),
            "issue": item.get("issue", ""),
            "pages": item.get("page", ""),
            "doi": item.get("DOI", ""),
            "publisher": item.get("publisher", ""),
            "type": item.get("type", ""),
            "url": f"https://doi.org/{item.get('DOI', '')}" if item.get("DOI") else "",
            "cited_by_count": item.get("is-referenced-by-count", 0) or 0,
        }

    # ─── OpenAlex ───────────────────────────────────────────────────

    async def search_openalex(self, entry: LiteratureEntry) -> dict | None:
        """Look up an entry in OpenAlex."""
        try:
            # With a DOI: direct lookup
            if entry.doi:
                doi = self.validate_doi(entry.doi)
                if doi:
                    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
                resp = await self._get("openalex", url)
                if resp.status_code == 200:
                    return self._parse_openalex_item(resp.json())

            # Search by title
            params = {"per_page": 3}
            if entry.title:
                params["search"] = entry.title
            if entry.year:
                params["filter"] = f"publication_year:{entry.year}"

            resp = await self._get("openalex",
                "https://api.openalex.org/works",
                params=params,
            )
            if resp.status_code != 200:
                return None

            results = resp.json().get("results", [])
            for item in results[:3]:
                parsed = self._parse_openalex_item(item)
                if parsed and self._is_likely_match(entry, parsed):
                    return parsed

            return None

        except Exception as e:
            logger.debug(f"OpenAlex error: {e}")
            return None

    @staticmethod
    def _parse_openalex_item(item: dict) -> dict | None:
        if not item:
            return None

        authors = []
        for a in item.get("authorships", []):
            name = a.get("author", {}).get("display_name", "")
            if name:
                authors.append(name)

        # Venue/Journal
        primary = item.get("primary_location", {}) or {}
        source = primary.get("source", {}) or {}
        journal = source.get("display_name", "")

        biblio = item.get("biblio", {}) or {}

        return {
            "source": "OpenAlex",
            "title": item.get("display_name", "") or item.get("title", ""),
            "authors": authors,
            "year": str(item.get("publication_year", "")),
            "journal": journal,
            "volume": biblio.get("volume", "") or "",
            "issue": biblio.get("issue", "") or "",
            "pages": (
                f"{biblio.get('first_page', '')}"
                f"{'-' + str(biblio['last_page']) if biblio.get('last_page') else ''}"
            ) if biblio.get("first_page") else "",
            "doi": (item.get("doi", "") or "").replace("https://doi.org/", ""),
            "publisher": "",
            "type": item.get("type", ""),
            "url": item.get("doi", "") or "",
            "cited_by_count": item.get("cited_by_count", 0) or 0,
        }

    # ─── Semantic Scholar ───────────────────────────────────────────

    async def search_semantic_scholar(self, entry: LiteratureEntry) -> dict | None:
        """Look up an entry in Semantic Scholar."""
        try:
            query = entry.title
            if not query:
                return None

            resp = await self._get("s2",
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": 3,
                    "fields": "title,authors,year,venue,externalIds,publicationVenue,citationCount",
                },
            )
            if resp.status_code != 200:
                return None

            data = resp.json().get("data", [])
            for item in data[:3]:
                parsed = self._parse_s2_item(item)
                if parsed and self._is_likely_match(entry, parsed):
                    return parsed

            return None

        except Exception as e:
            logger.debug(f"Semantic Scholar error: {e}")
            return None

    @staticmethod
    def _parse_s2_item(item: dict) -> dict | None:
        if not item:
            return None

        authors = [
            a.get("name", "") for a in item.get("authors", [])
            if a.get("name")
        ]

        ext_ids = item.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")

        venue = item.get("venue", "")
        if not venue:
            pv = item.get("publicationVenue", {}) or {}
            venue = pv.get("name", "")

        return {
            "source": "Semantic Scholar",
            "title": item.get("title", ""),
            "authors": authors,
            "year": str(item.get("year", "")),
            "journal": venue,
            "volume": "",
            "issue": "",
            "pages": "",
            "doi": doi,
            "publisher": "",
            "type": "",
            "url": f"https://doi.org/{doi}" if doi else "",
            "cited_by_count": item.get("citationCount", 0) or 0,
        }

    # ─── DBLP ──────────────────────────────────────────────────────

    async def search_dblp(self, entry: LiteratureEntry) -> dict | None:
        """Look up an entry in DBLP (computer science)."""
        try:
            query = entry.title
            if not query:
                return None

            resp = await self._get("dblp",
                "https://dblp.org/search/publ/api",
                params={"q": query, "format": "json", "h": 3},
            )
            if resp.status_code != 200:
                return None

            hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
            for hit in hits[:3]:
                info = hit.get("info", {})
                parsed = self._parse_dblp_item(info)
                if parsed and self._is_likely_match(entry, parsed):
                    return parsed

            return None

        except Exception as e:
            logger.debug(f"DBLP error: {e}")
            return None

    @staticmethod
    def _parse_dblp_item(info: dict) -> dict | None:
        if not info:
            return None

        def _s(val):
            """DBLP can return strings, dicts or lists for fields."""
            if val is None:
                return ""
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                return val.get("text", "") or str(val)
            if isinstance(val, list):
                return val[0] if val and isinstance(val[0], str) else ""
            return str(val)

        # Authors can be a string or a list
        authors_raw = info.get("authors", {}).get("author", [])
        if isinstance(authors_raw, str):
            authors = [authors_raw]
        elif isinstance(authors_raw, list):
            authors = [
                (a.get("text", "") if isinstance(a, dict) else str(a))
                for a in authors_raw
            ]
        else:
            authors = []

        title = _s(info.get("title", ""))
        if title.endswith("."):
            title = title[:-1]

        return {
            "source": "DBLP",
            "title": title,
            "authors": authors,
            "year": _s(info.get("year", "")),
            "journal": _s(info.get("venue", "")),
            "volume": _s(info.get("volume", "")),
            "issue": _s(info.get("number", "")),
            "pages": _s(info.get("pages", "")),
            "doi": _s(info.get("doi", "")),
            "publisher": "",
            "type": _s(info.get("type", "")),
            "url": _s(info.get("ee", "")) or _s(info.get("url", "")),
        }

    # ─── arXiv ──────────────────────────────────────────────────────

    # Regex for arXiv IDs: 2310.11511, 2401.08406, etc.
    _ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

    @classmethod
    def _extract_arxiv_id(cls, entry: LiteratureEntry) -> str | None:
        """Extract an arXiv ID from a DOI, URL or text."""
        # From a DOI: 10.48550/arxiv.2310.11511
        if entry.doi:
            m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", entry.doi, re.IGNORECASE)
            if m:
                return m.group(1)

        # From a URL: arxiv.org/abs/2310.11511
        if entry.url:
            m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", entry.url)
            if m:
                return m.group(1)

        # From raw_text or title
        for text in [entry.raw_text, entry.title]:
            if text:
                m = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5})", text, re.IGNORECASE)
                if m:
                    return m.group(1)

        return None

    @classmethod
    def _is_arxiv_paper(cls, entry: LiteratureEntry) -> bool:
        """Check whether an entry is probably an arXiv paper."""
        if cls._extract_arxiv_id(entry):
            return True
        for text in [entry.raw_text, entry.journal, entry.url, entry.doi]:
            if text and "arxiv" in str(text).lower():
                return True
        return False

    async def search_arxiv(self, entry: LiteratureEntry) -> dict | None:
        """Look up an entry in the arXiv API.

        arXiv API: http://export.arxiv.org/api/query
        Returns Atom XML with exact metadata.
        """
        try:
            # Step 1: direct lookup with a known ID
            arxiv_id = self._extract_arxiv_id(entry)
            if arxiv_id:
                resp = await self._get("arxiv",
                    "http://export.arxiv.org/api/query",
                    params={"id_list": arxiv_id, "max_results": 1},
                )
                if resp.status_code == 200:
                    result = self._parse_arxiv_response(resp.text)
                    if result:
                        return result

            # Step 2: title search
            if entry.title:
                # arXiv API: search by title
                query_parts = []
                # Clean the title: remove colons and special characters
                clean_title = re.sub(r"[^\w\s]", " ", entry.title)
                clean_title = re.sub(r"\s+", " ", clean_title).strip()
                query_parts.append(f'ti:"{clean_title[:100]}"')

                if entry.authors:
                    # family name of the first author
                    first_author = entry.authors[0].split()[-1]
                    query_parts.append(f"au:{first_author}")

                query = " AND ".join(query_parts)

                resp = await self._get("arxiv",
                    "http://export.arxiv.org/api/query",
                    params={
                        "search_query": query,
                        "max_results": 3,
                        "sortBy": "relevance",
                    },
                )
                if resp.status_code == 200:
                    result = self._parse_arxiv_response(resp.text)
                    if result and self._is_likely_match(entry, result):
                        return result

                # Fallback: broader search by title words only
                if not entry.title:
                    return None
                words = clean_title.split()[:6]
                fallback_query = " AND ".join(f"all:{w}" for w in words if len(w) > 3)
                if entry.authors:
                    fallback_query += f" AND au:{entry.authors[0].split()[-1]}"

                resp = await self._get("arxiv",
                    "http://export.arxiv.org/api/query",
                    params={
                        "search_query": fallback_query,
                        "max_results": 5,
                        "sortBy": "relevance",
                    },
                )
                if resp.status_code == 200:
                    # Several results → find the best match
                    results = self._parse_arxiv_response_multi(resp.text)
                    for r in results:
                        if self._is_likely_match(entry, r):
                            return r

            return None

        except Exception as e:
            logger.debug(f"arXiv error: {e}")
            return None

    def _parse_arxiv_response(self, xml_text: str) -> dict | None:
        """Parse an arXiv API response (first result)."""
        results = self._parse_arxiv_response_multi(xml_text)
        return results[0] if results else None

    @staticmethod
    def _parse_arxiv_response_multi(xml_text: str) -> list[dict]:
        """Parse an arXiv API response (all results)."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        for entry_el in root.findall("atom:entry", ns):
            title_el = entry_el.find("atom:title", ns)
            title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""

            if not title or title == "Error":
                continue

            # Authors
            authors = []
            for author_el in entry_el.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            # ID → extract the arXiv ID
            id_el = entry_el.find("atom:id", ns)
            arxiv_url = id_el.text.strip() if id_el is not None else ""
            arxiv_id = ""
            m = re.search(r"(\d{4}\.\d{4,5})", arxiv_url)
            if m:
                arxiv_id = m.group(1)

            # Datum
            published_el = entry_el.find("atom:published", ns)
            year = ""
            if published_el is not None and published_el.text:
                year = published_el.text[:4]

            # Category
            categories = []
            for cat_el in entry_el.findall("atom:category", ns):
                term = cat_el.get("term", "")
                if term:
                    categories.append(term)

            # DOI (if present in the links)
            doi = ""
            for link_el in entry_el.findall("atom:link", ns):
                href = link_el.get("href", "")
                if "doi.org" in href:
                    doi = href.replace("https://doi.org/", "").replace("http://doi.org/", "")

            if not doi and arxiv_id:
                doi = f"10.48550/arXiv.{arxiv_id}"

            results.append({
                "source": "arXiv",
                "title": title,
                "authors": authors,
                "year": year,
                "journal": f"arXiv:{arxiv_id}" if arxiv_id else "arXiv",
                "volume": "",
                "issue": "",
                "pages": "",
                "doi": doi,
                "publisher": "arXiv",
                "type": "preprint",
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else arxiv_url,
                "arxiv_id": arxiv_id,
                "categories": categories,
            })

        return results

    # ─── OpenLibrary (ISBN search, free) ─────────────────────

    async def search_openlibrary(self, entry: LiteratureEntry) -> dict | None:
        """Look up a book by ISBN in OpenLibrary (free, no API key)."""
        isbn = entry.isbn.strip().replace("-", "").replace(" ", "")
        if not isbn:
            return None

        try:
            url = "https://openlibrary.org/search.json"
            resp = await self._get("openlibrary", url, params={
                "isbn": isbn, "limit": "1",
            })
            if resp.status_code != 200:
                return None

            docs = resp.json().get("docs", [])
            if not docs:
                return None

            doc = docs[0]
            authors = doc.get("author_name", [])
            year = str(doc.get("first_publish_year", ""))
            title = doc.get("title", "")
            publisher_list = doc.get("publisher", [])

            parsed = {
                "title": title,
                "authors": authors,
                "year": year,
                "publisher": publisher_list[0] if publisher_list else "",
                "isbn": isbn,
                "source": "OpenLibrary",
                "url": f"https://openlibrary.org/isbn/{isbn}",
            }

            if self._is_likely_match(entry, parsed):
                return parsed
            return None

        except Exception as e:
            logger.debug(f"OpenLibrary error: {e}")
            return None

    # ─── Query-based search (for the analysis use cases) ──────────────
    #
    # In contrast to the search_* methods above, these methods take a
    # natural-language query and return up to `limit` SearchResult
    # objects. They are meant for discovering literature, not for
    # verifying an existing LiteratureEntry.
    #
    # Corresponding service class: src/connectors/literature_search.py

    async def query_openalex(
        self,
        query: str,
        limit: int = 25,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        peer_reviewed_only: bool = False,
    ) -> list:
        """Query-based search in OpenAlex.

        OpenAlex is the primary source: best coverage, stable API without
        mandatory authentication, has citation data and a peer-review
        signal.

        Returns:
            list[SearchResult] — the SearchResult class is imported lazily
            to avoid a circular import with literature_search.py.
        """

        try:
            params: dict = {
                "search": query,
                "per_page": min(limit, 50),  # OpenAlex maximum per page
            }
            filters = []
            if year_from:
                if year_to:
                    filters.append(f"publication_year:{year_from}-{year_to}")
                else:
                    filters.append(f"publication_year:>{year_from - 1}")
            elif year_to:
                filters.append(f"publication_year:<{year_to + 1}")
            if peer_reviewed_only:
                # Journal articles are the best approximation of peer review
                filters.append("primary_location.source.type:journal")
            if filters:
                params["filter"] = ",".join(filters)

            resp = await self._get(
                "openalex",
                "https://api.openalex.org/works",
                params=params,
            )
            if resp.status_code != 200:
                logger.warning(
                    f"OpenAlex query status {resp.status_code}: {query[:60]}"
                )
                return []

            items = resp.json().get("results", [])
            results: list = []
            for item in items[:limit]:
                sr = self._openalex_item_to_search_result(item)
                if sr:
                    results.append(sr)
            return results

        except Exception as e:
            logger.debug(f"OpenAlex query error: {e}")
            return []

    @staticmethod
    def _reconstruct_openalex_abstract(inverted_index: dict) -> str:
        """Reconstruct the abstract from OpenAlex' inverted_index.

        OpenAlex stores abstracts as {word: [positions]} to avoid
        full-text licensing problems. We can reconstruct the abstract as a
        string from it.
        """
        if not inverted_index or not isinstance(inverted_index, dict):
            return ""
        # Build a flat list of (position, word)
        positions: list[tuple[int, str]] = []
        for word, pos_list in inverted_index.items():
            if not isinstance(pos_list, list):
                continue
            for pos in pos_list:
                positions.append((pos, word))
        positions.sort()
        return " ".join(word for _, word in positions)

    def _openalex_item_to_search_result(self, item: dict):
        """Convert an OpenAlex work item into a SearchResult."""
        from src.connectors.literature_search import SearchResult

        if not item or not isinstance(item, dict):
            return None

        # Authors
        authors: list[str] = []
        for a in item.get("authorships", []) or []:
            name = (a.get("author") or {}).get("display_name", "")
            if name:
                authors.append(name)

        # Abstract
        abstract = self._reconstruct_openalex_abstract(
            item.get("abstract_inverted_index") or {}
        )

        # Venue / Journal
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        venue = source.get("display_name", "") or ""
        source_type = (source.get("type") or "").lower()

        # Peer-review signal: journal type as an approximation
        is_peer_reviewed: Optional[bool] = None
        if source_type == "journal":
            is_peer_reviewed = True
        elif source_type in ("repository", "preprint"):
            is_peer_reviewed = False

        # DOI and IDs
        doi = (item.get("doi") or "").replace("https://doi.org/", "").strip()
        openalex_id = item.get("id", "")

        # Year
        year_raw = item.get("publication_year") or 0
        try:
            year = int(year_raw) if year_raw else 0
        except (ValueError, TypeError):
            year = 0

        return SearchResult(
            id=openalex_id or doi,
            doi=doi,
            title=item.get("display_name", "") or item.get("title", ""),
            authors=authors,
            year=year,
            venue=venue,
            abstract=abstract,
            url=item.get("doi", "") or openalex_id or "",
            cited_by_count=int(item.get("cited_by_count") or 0),
            is_peer_reviewed=is_peer_reviewed,
            paper_type=item.get("type", "") or "",
            sources=["openalex"],
        )

    async def query_semantic_scholar(
        self,
        query: str,
        limit: int = 25,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> list:
        """Query-based search in Semantic Scholar.

        S2 delivers the best abstracts (NLP-enriched), but has stricter
        rate limits without an API key (1 request/s).
        """

        try:
            fields = (
                "title,authors,year,venue,externalIds,publicationVenue,"
                "citationCount,abstract,isOpenAccess,publicationTypes"
            )
            params: dict = {
                "query": query,
                "limit": min(limit, 100),
                "fields": fields,
            }
            if year_from or year_to:
                year_parts = []
                if year_from:
                    year_parts.append(str(year_from))
                else:
                    year_parts.append("")
                year_parts.append("-")
                if year_to:
                    year_parts.append(str(year_to))
                params["year"] = "".join(year_parts)

            resp = await self._get(
                "s2",
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
            )
            if resp.status_code != 200:
                logger.warning(
                    f"S2 query status {resp.status_code}: {query[:60]}"
                )
                return []

            data = resp.json().get("data", []) or []
            results: list = []
            for item in data[:limit]:
                sr = self._s2_item_to_search_result(item)
                if sr:
                    results.append(sr)
            return results

        except Exception as e:
            logger.debug(f"S2 query error: {e}")
            return []

    @staticmethod
    def _s2_item_to_search_result(item: dict):
        """Convert a Semantic Scholar paper into a SearchResult."""
        from src.connectors.literature_search import SearchResult

        if not item:
            return None

        authors = [
            a.get("name", "") for a in (item.get("authors") or [])
            if a.get("name")
        ]

        ext_ids = item.get("externalIds") or {}
        doi = (ext_ids.get("DOI") or "").strip()
        s2_id = item.get("paperId", "") or ""

        venue = item.get("venue", "") or ""
        if not venue:
            pv = item.get("publicationVenue") or {}
            venue = pv.get("name", "") or ""

        # Publication type (S2 returns a list such as ["JournalArticle"])
        pub_types = item.get("publicationTypes") or []
        is_peer_reviewed: Optional[bool] = None
        paper_type = ""
        if pub_types:
            pub_type_set = {t.lower() for t in pub_types}
            paper_type = pub_types[0]
            if "journalarticle" in pub_type_set:
                is_peer_reviewed = True
            elif "conference" in pub_type_set:
                is_peer_reviewed = True

        year_raw = item.get("year") or 0
        try:
            year = int(year_raw) if year_raw else 0
        except (ValueError, TypeError):
            year = 0

        return SearchResult(
            id=s2_id or doi,
            doi=doi,
            title=item.get("title", "") or "",
            authors=authors,
            year=year,
            venue=venue,
            abstract=item.get("abstract", "") or "",
            url=f"https://doi.org/{doi}" if doi else "",
            cited_by_count=int(item.get("citationCount") or 0),
            is_peer_reviewed=is_peer_reviewed,
            paper_type=paper_type,
            sources=["semantic_scholar"],
        )

    async def query_arxiv(
        self,
        query: str,
        limit: int = 25,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> list:
        """Query-based search in arXiv.

        arXiv delivers preprints — no citation data, no peer review, but
        current research in CS/physics/maths. Mostly irrelevant for the
        social sciences.

        Reuses the arXiv parser, extended by the abstract field (summary).
        """
        from src.connectors.literature_search import SearchResult

        try:
            # arXiv query syntax: all:"<term>" searches all fields
            # Clean up special characters
            clean_query = re.sub(r'[^\w\s]', ' ', query)
            clean_query = re.sub(r'\s+', ' ', clean_query).strip()
            if not clean_query:
                return []
            # Join several words with AND (arXiv syntax)
            words = [w for w in clean_query.split() if len(w) > 2]
            if not words:
                return []
            search_query = " AND ".join(f"all:{w}" for w in words[:8])

            resp = await self._get(
                "arxiv",
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": search_query,
                    "max_results": min(limit, 50),
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            if resp.status_code != 200:
                return []

            # Use the extended parser, which also returns the abstract
            parsed = self._parse_arxiv_response_full(resp.text)
            results: list = []
            for p in parsed:
                # Year filter as post-processing (the arXiv API has no filter parameter)
                paper_year = p.get("year", 0)
                if year_from and paper_year and paper_year < year_from:
                    continue
                if year_to and paper_year and paper_year > year_to:
                    continue

                results.append(SearchResult(
                    id=p.get("arxiv_id", "") or p.get("doi", ""),
                    doi=p.get("doi", ""),
                    title=p.get("title", ""),
                    authors=p.get("authors", []),
                    year=paper_year,
                    venue=p.get("venue", "arXiv"),
                    abstract=p.get("abstract", ""),
                    url=p.get("url", ""),
                    cited_by_count=0,  # arXiv has no citation data
                    is_peer_reviewed=False,
                    paper_type="preprint",
                    sources=["arxiv"],
                ))
                if len(results) >= limit:
                    break
            return results

        except Exception as e:
            logger.debug(f"arXiv query error: {e}")
            return []

    @staticmethod
    def _parse_arxiv_response_full(xml_text: str) -> list[dict]:
        """Extended arXiv parser with abstract (summary tag).

        _parse_arxiv_response_multi returns no abstract, but the query
        search needs it. This parser is a superset: it returns all fields
        of the other plus abstract and year.
        """
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        for entry_el in root.findall("atom:entry", ns):
            title_el = entry_el.find("atom:title", ns)
            title = ""
            if title_el is not None and title_el.text:
                title = title_el.text.strip().replace("\n", " ")
                title = re.sub(r"\s+", " ", title)

            if not title or title == "Error":
                continue

            authors: list[str] = []
            for author_el in entry_el.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            id_el = entry_el.find("atom:id", ns)
            arxiv_url = id_el.text.strip() if id_el is not None else ""
            arxiv_id = ""
            m = re.search(r"(\d{4}\.\d{4,5})", arxiv_url)
            if m:
                arxiv_id = m.group(1)

            published_el = entry_el.find("atom:published", ns)
            year = 0
            if published_el is not None and published_el.text:
                try:
                    year = int(published_el.text[:4])
                except (ValueError, TypeError):
                    year = 0

            # Abstract (summary-Tag)
            summary_el = entry_el.find("atom:summary", ns)
            abstract = ""
            if summary_el is not None and summary_el.text:
                abstract = summary_el.text.strip().replace("\n", " ")
                abstract = re.sub(r"\s+", " ", abstract)

            doi = ""
            for link_el in entry_el.findall("atom:link", ns):
                href = link_el.get("href", "") or ""
                if "doi.org" in href:
                    doi = href.replace("https://doi.org/", "").replace(
                        "http://doi.org/", ""
                    )
            if not doi and arxiv_id:
                doi = f"10.48550/arXiv.{arxiv_id}"

            results.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "doi": doi,
                "venue": f"arXiv:{arxiv_id}" if arxiv_id else "arXiv",
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else arxiv_url,
            })

        return results

    # ─── Citation Chasing ──────────────────────────────────────────

    async def get_citers_openalex(
        self,
        doi: str,
        limit: int = 25,
    ) -> list:
        """Forward citation: all papers that cite the given DOI.

        Uses OpenAlex' `cites:` filter — format: cites:<openalex_work_id>.
        The work ID has to be resolved from the DOI first.
        """

        try:
            # Step 1: resolve DOI → OpenAlex work ID
            doi_clean = doi.strip().replace("https://doi.org/", "")
            resp = await self._get(
                "openalex",
                f"https://api.openalex.org/works/https://doi.org/{doi_clean}",
            )
            if resp.status_code != 200:
                logger.debug(
                    f"OpenAlex DOI-Lookup {doi_clean}: {resp.status_code}"
                )
                return []

            work = resp.json()
            work_id = work.get("id", "")
            if not work_id:
                return []
            # work_id is a URL like https://openalex.org/W1234 → extract the W ID
            m = re.search(r"W\d+", work_id)
            if not m:
                return []
            short_id = m.group(0)

            # Step 2: all works that cite this work
            resp2 = await self._get(
                "openalex",
                "https://api.openalex.org/works",
                params={
                    "filter": f"cites:{short_id}",
                    "per_page": min(limit, 50),
                    "sort": "cited_by_count:desc",
                },
            )
            if resp2.status_code != 200:
                return []

            items = resp2.json().get("results", []) or []
            results: list = []
            for item in items[:limit]:
                sr = self._openalex_item_to_search_result(item)
                if sr:
                    results.append(sr)
            return results

        except Exception as e:
            logger.debug(f"OpenAlex get_citers error for {doi}: {e}")
            return []

    async def get_references_openalex(
        self,
        doi: str,
        limit: int = 25,
    ) -> list:
        """Backward citation: all papers this paper cites.

        Uses OpenAlex' `referenced_works` field. Every reference is a work
        ID; the metadata is fetched in a second call with an OR filter
        (OpenAlex allows up to ~50 IDs per OR filter).
        """

        try:
            doi_clean = doi.strip().replace("https://doi.org/", "")
            resp = await self._get(
                "openalex",
                f"https://api.openalex.org/works/https://doi.org/{doi_clean}",
            )
            if resp.status_code != 200:
                return []

            work = resp.json()
            referenced = work.get("referenced_works", []) or []
            if not referenced:
                return []

            # Extract work IDs (URL → short)
            short_ids: list[str] = []
            for ref_url in referenced:
                m = re.search(r"W\d+", ref_url)
                if m:
                    short_ids.append(m.group(0))
            if not short_ids:
                return []

            # Fetch metadata in batches (OpenAlex OR-filter limit: ~50)
            results: list = []
            batch_size = 40
            remaining = limit
            for i in range(0, len(short_ids), batch_size):
                if remaining <= 0:
                    break
                batch = short_ids[i:i + batch_size]
                or_filter = "|".join(batch)
                resp2 = await self._get(
                    "openalex",
                    "https://api.openalex.org/works",
                    params={
                        "filter": f"openalex_id:{or_filter}",
                        "per_page": min(remaining, batch_size),
                    },
                )
                if resp2.status_code != 200:
                    continue
                items = resp2.json().get("results", []) or []
                for item in items:
                    sr = self._openalex_item_to_search_result(item)
                    if sr:
                        results.append(sr)
                        remaining -= 1
                        if remaining <= 0:
                            break
            # Sort by citations
            results.sort(key=lambda r: r.cited_by_count, reverse=True)
            return results[:limit]

        except Exception as e:
            logger.debug(f"OpenAlex get_references error for {doi}: {e}")
            return []

    # ─── Parallel search ───────────────────────────────────────────

    async def lookup_entry(self, entry: LiteratureEntry) -> list[dict]:
        """Look up an entry in ALL databases in parallel.

        arXiv papers are looked up preferentially via the arXiv API.
        Books with an ISBN are additionally looked up in OpenLibrary.
        """
        tasks = [
            self.search_crossref(entry),
            self.search_openalex(entry),
            self.search_semantic_scholar(entry),
            self.search_dblp(entry),
        ]

        # ISBN present → also ask OpenLibrary
        if entry.isbn:
            tasks.append(self.search_openlibrary(entry))

        # arXiv papers: additionally query the arXiv API (preferred)
        if self._is_arxiv_paper(entry):
            tasks.insert(0, self.search_arxiv(entry))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        matches = []
        for r in results:
            if isinstance(r, dict) and r:
                matches.append(r)
            elif isinstance(r, Exception):
                logger.debug(f"API error: {r}")

        # Sort the arXiv result first (if present)
        matches.sort(key=lambda m: (m.get("source") != "arXiv", m.get("source", "")))

        return matches

    # ─── Matching heuristic ────────────────────────────────────────

    @staticmethod
    def _normalize_title(title) -> str:
        """Normalise titles for comparison."""
        if not isinstance(title, str):
            if isinstance(title, list):
                title = title[0] if title else ""
            elif isinstance(title, dict):
                title = title.get("text", "") or ""
            else:
                title = str(title) if title else ""
        t = title.lower().strip()
        t = re.sub(r"[^\w\s]", "", t)  # remove punctuation
        t = re.sub(r"\s+", " ", t)
        return t

    @classmethod
    def _is_likely_match(cls, entry: LiteratureEntry, result: dict) -> bool:
        """Check whether an API result matches the entry."""
        # Direct DOI match: if both DOIs agree → certain
        if entry.doi and result.get("doi"):
            d1 = entry.doi.strip().lower().removeprefix("https://doi.org/")
            d2 = result["doi"].strip().lower().removeprefix("https://doi.org/")
            if d1 and d2 and d1 == d2:
                return True

        if not entry.title or not result.get("title"):
            return False

        # Title similarity
        t1 = cls._normalize_title(entry.title)
        t2 = cls._normalize_title(result.get("title", ""))
        similarity = SequenceMatcher(None, t1, t2).ratio()

        if similarity >= 0.80:
            return True

        # Subtitle tolerance: one title is a substring of the other
        if len(t1) > 20 and len(t2) > 20:
            if t1 in t2 or t2 in t1:
                return True

        # With lower title similarity: check author + year
        if similarity >= 0.55:
            year_match = (
                not entry.year
                or not result.get("year")
                or entry.year == result.get("year")
            )
            author_match = cls._authors_overlap(
                entry.authors, result.get("authors", [])
            )
            if year_match and author_match:
                return True

        return False

    @staticmethod
    def _authors_overlap(authors1: list, authors2: list) -> bool:
        """Check whether at least one author occurs in both lists."""
        if not authors1 or not authors2:
            return True

        def last_names(authors):
            names = set()
            for a in authors:
                if not isinstance(a, str):
                    a = a.get("text", "") if isinstance(a, dict) else str(a)
                a = a.strip().rstrip(".,")
                if not a:
                    continue
                if "," in a:
                    # "Family, Given" → last word before the comma
                    before = a.split(",")[0].strip()
                    parts = before.split()
                else:
                    parts = a.split()
                if parts:
                    last = parts[-1].rstrip(".")
                    # initial (≤2 characters) → second-to-last word
                    if len(last) <= 2 and len(parts) >= 2:
                        names.add(parts[-2].lower().rstrip(".,"))
                    else:
                        names.add(last.lower())
            return names

        set1 = last_names(authors1)
        set2 = last_names(authors2)
        return bool(set1 & set2)

    @staticmethod
    def validate_doi(doi: str) -> str | None:
        """Validate and normalise a DOI.

        Returns: the normalised DOI (without URL prefix) or None if invalid.
        DOI format: 10.XXXX/YYYY (registrant ID / suffix)
        """
        if not doi:
            return None
        doi = doi.strip()
        # Remove the URL prefix
        for prefix in [
            "https://doi.org/", "http://doi.org/",
            "https://dx.doi.org/", "http://dx.doi.org/",
            "doi:", "DOI:",
        ]:
            if doi.lower().startswith(prefix.lower()):
                doi = doi[len(prefix):]
                break
        doi = doi.strip()
        # Check the format: 10.XXXX/...
        if not re.match(r"^10\.\d{4,}/\S+$", doi):
            return None
        return doi

    # ─── Field comparison ────────────────────────────────────────────

    @classmethod
    def compare_fields(
        cls, entry: LiteratureEntry, match: dict,
    ) -> list[dict]:
        """Compare fields between the original entry and the API match.

        Returns: list of deviations.
        """
        deviations = []

        def _to_str(val) -> str:
            if val is None:
                return ""
            if isinstance(val, str):
                # Decode HTML entities: &amp; → &, etc.
                from html import unescape
                return unescape(val)
            if isinstance(val, list):
                from html import unescape
                return unescape(", ".join(str(x) for x in val))
            if isinstance(val, dict):
                return val.get("text", "") or val.get("name", "") or str(val)
            return str(val)

        def _normalize(text: str) -> str:
            """Normalise typographic variants before the comparison."""
            text = text.strip().lower()
            # Unicode dashes → plain hyphen
            text = text.replace("\u2013", "-")  # en dash –
            text = text.replace("\u2014", "-")  # em dash —
            text = text.replace("\u2012", "-")  # figure dash ‒
            text = text.replace("\u2015", "-")  # horizontal bar ―
            # Typographic quotation marks → plain ones
            text = text.replace("\u201c", '"').replace("\u201d", '"')
            text = text.replace("\u2018", "'").replace("\u2019", "'")
            # Remove a trailing full stop/comma
            text = text.rstrip(".,;")
            # Spaces around hyphens in page ranges: "31 - 68" → "31-68"
            text = re.sub(r"\s*-\s*", "-", text)
            # Multiple spaces
            text = re.sub(r"\s+", " ", text)
            return text

        def _normalize_doi(doi: str) -> str:
            """Normalise a DOI: prefix, case."""
            doi = doi.strip().lower()
            doi = doi.removeprefix("https://doi.org/")
            doi = doi.removeprefix("http://doi.org/")
            doi = doi.removeprefix("doi:")
            return doi

        def _normalize_journal(journal: str) -> str:
            """Normalise journal names for comparison.

            Removes parenthesised additions such as (Springer), (Cornell
            University), and typical suffixes.
            """
            j = journal.strip().lower()
            # Remove parenthesised additions: "Nature (Springer)" → "nature"
            j = re.sub(r"\s*\([^)]*\)\s*$", "", j)
            # trailing full stop
            j = j.rstrip(".")
            return j

        def _check(field_name: str, display_name: str,
                    original, found,
                    fuzzy: bool = False):
            original = _to_str(original)
            found = _to_str(found)

            if not original and not found:
                return
            if not original:
                # Missing vol/issue/pages are not relevant errors
                if field_name in ("volume", "issue", "pages"):
                    return
                # A missing journal is normal for books/book chapters
                if field_name == "journal" and entry.entry_type in (
                    "book", "inproceedings", "thesis", "other",
                ):
                    return
                deviations.append({
                    "field": field_name,
                    "display_name": display_name,
                    "type": "missing_in_original",
                    "original": "",
                    "found": found,
                    "source": match.get("source", ""),
                    "message": tc("lit.dev.missing_in_original", field=display_name, found=found),
                })
                return
            if not found:
                return

            # Field-specific normalisation
            if field_name == "doi":
                orig_clean = _normalize_doi(original)
                found_clean = _normalize_doi(found)
            elif field_name == "journal":
                orig_clean = _normalize_journal(original)
                found_clean = _normalize_journal(found)
            else:
                orig_clean = _normalize(original)
                found_clean = _normalize(found)

            if orig_clean == found_clean:
                return

            # Fuzzy comparison for title and journal
            if fuzzy:
                sim = SequenceMatcher(None, orig_clean, found_clean).ratio()
                if sim >= 0.85:
                    return
                # Journal: check whether one string is contained in the other:
                # "Artificial Intelligence" is contained in
                # "artificial intelligence research"
                if field_name == "journal":
                    if (orig_clean in found_clean or found_clean in orig_clean):
                        return

            deviations.append({
                "field": field_name,
                "display_name": display_name,
                "type": "mismatch",
                "original": original,
                "found": found,
                "source": match.get("source", ""),
                "message": f'{display_name}: "{original}" -> "{found}"',
            })

        _check("title", tc("lit.field.title"), entry.title, match.get("title", ""),
               fuzzy=True)
        _check("year", tc("lit.field.year"), entry.year, match.get("year", ""))
        _check("journal", "Journal/Venue", entry.journal,
               match.get("journal", ""), fuzzy=True)
        _check("volume", tc("lit.field.volume"), entry.volume, match.get("volume", ""))
        _check("issue", tc("lit.field.issue"), entry.issue, match.get("issue", ""))
        _check("pages", tc("lit.field.pages"), entry.pages, match.get("pages", ""))
        _check("doi", "DOI", entry.doi, match.get("doi", ""))

        # ── Author comparison (based on family names) ─────────────────
        if entry.authors and match.get("authors"):
            orig_authors = [_to_str(a) for a in entry.authors if _to_str(a)]
            found_authors = [_to_str(a) for a in match["authors"] if _to_str(a)]

            def _extract_last_name(name: str) -> str:
                """Extract the core family name from various formats.

                'Kirsh, D.' → 'kirsh'
                'David Kirsh' → 'kirsh'
                'Okafor C.' → 'okafor' (C. is an initial)
                'de Luis Balaguer, A.' → 'balaguer'
                """
                name = name.strip().rstrip(".,")
                if not name:
                    return ""
                if "," in name:
                    # "Family, Given" → last word before the comma
                    before_comma = name.split(",")[0].strip()
                    parts = before_comma.split()
                    return parts[-1].lower() if parts else ""
                # "Given Family" or "Family G."
                parts = name.split()
                if not parts:
                    return ""
                last = parts[-1].rstrip(".")
                # If the last word has ≤ 2 characters → probably an initial
                # → the second-to-last word is the family name
                if len(last) <= 2 and len(parts) >= 2:
                    return parts[-2].lower().rstrip(".,")
                return last.lower()

            orig_names = {_extract_last_name(a) for a in orig_authors}
            found_names = {_extract_last_name(a) for a in found_authors}
            # Remove empty strings
            orig_names.discard("")
            found_names.discard("")

            # If all family names agree and the count is equal:
            # → NO error (only a formatting difference like "D. Kirsh" vs "David Kirsh")
            if orig_names == found_names and len(orig_authors) == len(found_authors):
                pass  # no problem

            elif len(orig_authors) != len(found_authors):
                # Different count → report it,
                # but only if the names differ too
                missing = orig_names - found_names
                extra = found_names - orig_names
                if missing or extra:
                    deviations.append({
                        "field": "authors",
                        "display_name": tc("lit.field.author_count"),
                        "type": "mismatch",
                        "original": tc("lit.dev.n_authors", n=len(orig_authors)),
                        "found": (
                            tc("lit.dev.n_authors_named", n=len(found_authors), names=', '.join(found_authors[:5]))
                        ),
                        "source": match.get("source", ""),
                        "message": (
                            tc("lit.dev.author_count", orig=len(orig_authors), found=len(found_authors), source=match.get('source', '?'))
                        ),
                    })
            else:
                # Same count, but the family names differ
                missing = orig_names - found_names
                if missing:
                    deviations.append({
                        "field": "authors",
                        "display_name": tc("lit.field.author_names"),
                        "type": "mismatch",
                        "original": ", ".join(orig_authors[:5]),
                        "found": ", ".join(found_authors[:5]),
                        "source": match.get("source", ""),
                        "message": (
                            tc("lit.dev.author_names", orig=', '.join(orig_authors[:5]), source=match.get('source', '?'), found=', '.join(found_authors[:5]))
                        ),
                    })

        return deviations
