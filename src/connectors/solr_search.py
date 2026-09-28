"""
Solr connector for searching the institution's website.

Searches two Solr cores (one per language, default names
web-search_de and web-search_en) that index the institution's web content.
"""

import logging
import os
from dataclasses import dataclass

import httpx

from src.connectors.base import BaseConnector
from src.pipeline.models import SearchResult, SourceDocument, SourceType
from src.core.tls import tls_verify

logger = logging.getLogger(__name__)


@dataclass
class SolrConfig:
    """Configuration of the Solr connector."""
    host: str = ""
    scheme: str = "https"
    port: int = 443
    username: str = ""
    password: str = ""
    core_de: str = "web-search_de"
    core_en: str = "web-search_en"
    # Solr field names (adjustable if the index is structured differently)
    field_title: str = "title"
    field_content: str = "content"
    field_url: str = "url"
    field_description: str = "description"
    max_results: int = 15
    timeout: float = 30.0
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "SolrConfig":
        host = os.getenv("SOLR_HOST", "")
        return cls(
            host=host,
            scheme=os.getenv("SOLR_SCHEME", "https"),
            port=int(os.getenv("SOLR_PORT", "443")),
            username=os.getenv("SOLR_USERNAME", ""),
            password=os.getenv("SOLR_PASSWORD", ""),
            core_de=os.getenv("SOLR_CORE_DE", "web-search_de"),
            core_en=os.getenv("SOLR_CORE_EN", "web-search_en"),
            enabled=bool(host),
        )

    @property
    def base_url(self) -> str:
        port_str = f":{self.port}" if self.port not in (80, 443) else ""
        return f"{self.scheme}://{self.host}{port_str}/solr"


class SolrConnector(BaseConnector):
    """Search the Solr indices of the institution's website.

    Supports two cores: one for German and one for English content.
    Used primarily in the institution mode.
    """

    name = "solr"

    def __init__(self, config: SolrConfig):
        self.config = config
        auth = None
        if config.username and config.password:
            auth = (config.username, config.password)
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            auth=auth,
            verify=tls_verify(),
            follow_redirects=True,
        )

    async def search(
        self,
        query: str,
        max_results: int = 15,
        language: str | None = None,
    ) -> list[SearchResult]:
        """Search Solr — picks the core by language.

        Args:
            query: search term
            max_results: maximum number of results
            language: "de" → core_de, "en" → core_en,
                      None → search both cores
        """
        max_results = min(max_results, self.config.max_results)

        if language == "de":
            cores = [self.config.core_de]
        elif language == "en":
            cores = [self.config.core_en]
        else:
            # Search both cores
            cores = [self.config.core_de, self.config.core_en]

        all_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for core in cores:
            try:
                results = await self._search_core(
                    core, query,
                    max_results=max_results,
                )
                for r in results:
                    # De-duplicate by URL
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        all_results.append(r)
            except Exception as e:
                logger.warning(f"Solr search in {core} failed: {e}")

        logger.info(
            f"Solr: '{query}' → {len(all_results)} results "
            f"(Cores: {', '.join(cores)})"
        )
        return all_results[:max_results]

    async def _search_core(
        self,
        core: str,
        query: str,
        max_results: int = 15,
    ) -> list[SearchResult]:
        """Search a single Solr core."""
        url = f"{self.config.base_url}/{core}/select"

        # Solr query: search in title and content
        # DisMax/eDisMax for relevance-based search
        params = {
            "q": query,
            "wt": "json",
            "rows": max_results,
            "defType": "edismax",
            "qf": f"{self.config.field_title}^3 "
                  f"{self.config.field_content}^1 "
                  f"{self.config.field_description}^2",
            "fl": f"{self.config.field_title},"
                  f"{self.config.field_url},"
                  f"{self.config.field_content},"
                  f"{self.config.field_description},"
                  f"score",
            "hl": "true",
            "hl.fl": f"{self.config.field_content}",
            "hl.snippets": "2",
            "hl.fragsize": "300",
        }

        resp = await self.client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        response = data.get("response", {})
        docs = response.get("docs", [])
        highlighting = data.get("highlighting", {})

        results = []
        for doc in docs:
            # Solr fields can be arrays
            title = self._get_field(doc, self.config.field_title)
            doc_url = self._get_field(doc, self.config.field_url)
            description = self._get_field(doc, self.config.field_description)

            if not doc_url:
                continue

            # Snippet: prefer highlighting, otherwise the description
            doc_id = doc.get("id", "")
            hl = highlighting.get(doc_id, {})
            hl_snippets = hl.get(self.config.field_content, [])
            snippet = " ... ".join(hl_snippets) if hl_snippets else description

            results.append(SearchResult(
                title=title or doc_url,
                url=doc_url,
                snippet=snippet[:500] if snippet else "",
                source_type=SourceType.WEB_SEARCH,
                connector_name=f"solr:{core}",
            ))

        return results

    async def fetch(self, url: str) -> SourceDocument:
        """Try to load a URL's content from Solr.

        If the URL is in the index, the stored content is returned —
        faster than crawling the web.
        """
        for core in [self.config.core_de, self.config.core_en]:
            try:
                solr_url = f"{self.config.base_url}/{core}/select"
                params = {
                    "q": f'{self.config.field_url}:"{url}"',
                    "wt": "json",
                    "rows": 1,
                    "fl": f"{self.config.field_title},"
                          f"{self.config.field_url},"
                          f"{self.config.field_content}",
                }
                resp = await self.client.get(solr_url, params=params)
                resp.raise_for_status()
                data = resp.json()

                docs = data.get("response", {}).get("docs", [])
                if docs:
                    doc = docs[0]
                    title = self._get_field(doc, self.config.field_title)
                    content = self._get_field(doc, self.config.field_content)
                    if content and len(content) > 50:
                        return SourceDocument(
                            source_type=SourceType.WEB_PAGE,
                            url=url,
                            title=title or url,
                            content=content,
                        )
            except Exception as e:
                logger.debug(f"Solr fetch {core}/{url}: {e}")

        raise ValueError(f"URL not in the Solr index: {url}")

    def can_handle(self, url: str) -> bool:
        """Solr can deliver URLs of the institution's own website from the index."""
        from src.institution import get_profile
        return get_profile().matches_url(url)

    async def check_connectivity(self) -> bool:
        """Check whether Solr is reachable."""
        try:
            url = f"{self.config.base_url}/{self.config.core_de}/admin/ping"
            resp = await self.client.get(url, timeout=5.0)
            return resp.status_code == 200
        except Exception:
            # Fallback: simple search
            try:
                url = f"{self.config.base_url}/{self.config.core_de}/select"
                resp = await self.client.get(
                    url, params={"q": "*:*", "rows": "0", "wt": "json"},
                    timeout=5.0,
                )
                return resp.status_code == 200
            except Exception:
                return False

    async def verify_fields(self) -> dict:
        """Check whether the configured field names exist in the Solr index.

        Returns:
            {"ok": True/False, "found": [...], "missing": [...],
             "available": [...]}
        """
        expected = {
            self.config.field_title,
            self.config.field_content,
            self.config.field_url,
            self.config.field_description,
        }
        try:
            url = f"{self.config.base_url}/{self.config.core_de}/select"
            resp = await self.client.get(
                url,
                params={"q": "*:*", "rows": "1", "wt": "json"},
                timeout=10.0,
            )
            resp.raise_for_status()
            docs = resp.json().get("response", {}).get("docs", [])
            if not docs:
                return {"ok": False, "error": "No document in the index"}

            available = set(docs[0].keys())
            found = expected & available
            missing = expected - available

            if missing:
                logger.warning(
                    f"⚠️ Solr: configured fields missing in the index: "
                    f"{', '.join(sorted(missing))}. "
                    f"Available fields: {', '.join(sorted(available))}"
                )
            else:
                logger.info(
                    f"✓ Solr: all fields verified "
                    f"({', '.join(sorted(found))})"
                )

            return {
                "ok": len(missing) == 0,
                "found": sorted(found),
                "missing": sorted(missing),
                "available": sorted(available),
            }
        except Exception as e:
            logger.warning(f"Solr field check failed: {e}")
            return {"ok": False, "error": str(e)}

    @staticmethod
    def _get_field(doc: dict, field_name: str) -> str:
        """Extract a field from a Solr document (may be an array or a string)."""
        value = doc.get(field_name, "")
        if isinstance(value, list):
            return value[0] if value else ""
        return str(value) if value else ""

    async def close(self):
        await self.client.aclose()
