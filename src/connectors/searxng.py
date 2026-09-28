"""
SearXNG meta search engine connector.
"""

import logging
import httpx

from src.connectors.base import BaseConnector
from src.pipeline.models import SearchResult, SourceDocument, SourceType
from src.config import SearXNGConfig

logger = logging.getLogger(__name__)


# ─── Academic engine allow list ────────────────────────────────
#
# Used by the academic_only mode. The list covers context
# (Wikipedia, Wikidata) as well as scholarly publications (Google Scholar,
# Semantic Scholar, ArXiv, PubMed).
#
# Which engines are actually available depends on the SearXNG
# instance (see its settings.yml). Engines from this list that are not
# enabled in the instance are silently ignored by SearXNG — no harm done.
#
#
# Candidates for extension (disabled in a default SearXNG, but valuable
# for academic research): openalex, crossref. Enable them in the
# instance's settings.yml, then add them here.
ACADEMIC_ENGINES_WHITELIST = [
    "wikipedia",
    "wikidata",
    "google scholar",
    "semantic scholar",
    "arxiv",
    "pubmed",
]

ACADEMIC_ENGINES_STRING = ",".join(ACADEMIC_ENGINES_WHITELIST)


class SearXNGConnector(BaseConnector):
    """Connector for the local SearXNG instance."""

    name = "searxng"

    def __init__(self, config: SearXNGConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            follow_redirects=True,
        )

    async def search(
        self,
        query: str,
        max_results: int = 10,
        language: str | None = None,
        time_range: str | None = None,
        categories: str | None = None,
        engines: str | None = None,
    ) -> list[SearchResult]:
        """Run a meta search via SearXNG.

        Args:
            query: search term
            max_results: maximum number of results
            language: language code for SearXNG (e.g. "de", "en", "de-DE").
                      Default: "de-DE" (from the config or fallback).
            time_range: time range filter. SearXNG supports:
                        "day", "week", "month", "year".
                        Default: None (no time filter).
            categories: SearXNG categories, comma-separated,
                        e.g. "general", "science", "news", "it".
                        Default: "general" — ignored if `engines` is set.
            engines: explicit allow list of engine names, comma-separated,
                     e.g. "wikipedia,wikidata,google scholar,arxiv".
                     If set, `categories` is not sent, because SearXNG
                     treats both as a double condition (both must match →
                     often empty results).
                     Default: None — then `categories` is used.
        """
        max_results = min(max_results, self.config.max_results)

        # Language: parameter > config default > "de-DE"
        lang = language or getattr(self.config, "language", "de-DE")

        params = {
            "q": query,
            "format": "json",
            "pageno": 1,
            "language": lang,
        }

        # Either engines OR categories — not both.
        # SearXNG accepts both parameters at once, but then an engine must
        # satisfy BOTH (be in the category AND in the allow list), which with
        # academic allow lists often leads to unintentionally empty
        # results.
        if engines:
            params["engines"] = engines
        else:
            params["categories"] = categories or "general"

        # Only set the time range if given and valid
        valid_time_ranges = {"day", "week", "month", "year"}
        if time_range and time_range in valid_time_ranges:
            params["time_range"] = time_range

        try:
            resp = await self.client.get(
                f"{self.config.base_url}/search",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for r in data.get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    source_type=SourceType.WEB_SEARCH,
                    connector_name=self.name,
                ))

            tr_info = f", time_range={time_range}" if time_range else ""
            src_info = (
                f", engines={engines}" if engines
                else (f", cat={categories}" if categories and categories != "general" else "")
            )
            logger.info(
                f"SearXNG: '{query}' ({lang}{tr_info}{src_info}) "
                f"→ {len(results)} results"
            )
            return results

        except httpx.TimeoutException:
            logger.warning(f"SearXNG timeout for: '{query}'")
            return []
        except Exception as e:
            logger.error(f"SearXNG error: {e}")
            return []

    async def check_connectivity(self) -> bool:
        """Check whether SearXNG is reachable."""
        try:
            resp = await self.client.get(
                f"{self.config.base_url}/",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def fetch(self, url: str) -> SourceDocument:
        """SearXNG does not fetch itself — delegates to the WebScraper."""
        raise NotImplementedError(
            "SearXNG only searches; use WebScraper for fetching"
        )

    def can_handle(self, url: str) -> bool:
        return False  # search only, no fetch

    async def close(self):
        await self.client.aclose()
