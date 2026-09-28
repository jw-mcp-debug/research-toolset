"""
Elasticsearch connector for the research pipeline.
Searches an Elasticsearch index (e.g. website content) and returns
results as SearchResult / SourceDocument.
"""

import logging

import httpx

from src.connectors.base import BaseConnector, clean_url
from src.config import ElasticsearchConfig
from src.pipeline.models import SearchResult, SourceDocument, SourceType

logger = logging.getLogger(__name__)


class ElasticsearchConnector(BaseConnector):
    """Connector for Elasticsearch indices."""

    name = "elasticsearch"

    def __init__(self, config: ElasticsearchConfig):
        self.config = config
        self._base_url = config.base_url.rstrip("/")
        self._index = config.index
        self._site_base_url = config.site_base_url.rstrip("/") if config.site_base_url else ""

        # HTTP client with auth
        headers = {"Content-Type": "application/json"}
        auth = None

        if config.api_key:
            headers["Authorization"] = f"ApiKey {config.api_key}"
        elif config.username and config.password:
            auth = (config.username, config.password)

        self.client = httpx.AsyncClient(
            headers=headers,
            auth=auth,
            timeout=30.0,
            verify=True,
        )

    async def search(self, query: str, max_results: int = 10, **kwargs) -> list[SearchResult]:
        """Full-text search in the Elasticsearch index."""
        max_results = min(max_results, self.config.max_results)

        # Search fields
        search_fields = [self.config.field_title + "^2"]  # weight the title higher
        if self.config.field_body:
            search_fields.append(self.config.field_body)
        if self.config.field_path:
            search_fields.append(self.config.field_path)

        body = {
            "size": max_results,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": search_fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
            "_source": [
                self.config.field_title,
                self.config.field_url,
                self.config.field_body,
            ],
            "highlight": {
                "fields": {
                    self.config.field_body: {
                        "fragment_size": 200,
                        "number_of_fragments": 2,
                    }
                },
                "pre_tags": [""],
                "post_tags": [""],
            },
        }

        # Optional fields only when configured
        if self.config.field_path:
            body["_source"].append(self.config.field_path)

        try:
            url = f"{self._base_url}/{self._index}/_search"
            resp = await self.client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for hit in data.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})

                title = source.get(self.config.field_title, "")
                page_url = source.get(self.config.field_url, "")

                # Derive the URL from the path when there is no explicit URL field
                if not page_url and self.config.field_path and self._site_base_url:
                    path = source.get(self.config.field_path, "")
                    if path:
                        page_url = f"{self._site_base_url}/{path.lstrip('/')}"

                if not page_url:
                    continue

                # Snippet from highlight or body
                highlight = hit.get("highlight", {})
                body_highlights = highlight.get(self.config.field_body, [])
                if body_highlights:
                    snippet = " … ".join(body_highlights)
                else:
                    body_text = source.get(self.config.field_body, "")
                    snippet = body_text[:300] if body_text else ""

                results.append(SearchResult(
                    title=title or page_url,
                    url=clean_url(page_url),
                    snippet=snippet,
                    source_type=SourceType.ELASTIC,
                    connector_name=self.name,
                ))

            logger.info(f"Elasticsearch: '{query}' → {len(results)} results")
            return results

        except httpx.HTTPStatusError as e:
            logger.warning(f"Elasticsearch HTTP error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.warning(f"Elasticsearch search failed: {e}")
            return []

    async def fetch(self, url: str) -> SourceDocument:
        """Load the full content of a page from the index."""
        # Look up the exact URL in the index
        body = {
            "size": 1,
            "query": {
                "term": {
                    self.config.field_url: url
                }
            },
        }

        # Fallback: also search by path
        if self.config.field_path and self._site_base_url:
            path = url.replace(self._site_base_url, "").lstrip("/")
            body = {
                "size": 1,
                "query": {
                    "bool": {
                        "should": [
                            {"term": {self.config.field_url: url}},
                            {"term": {self.config.field_path: path}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }

        try:
            search_url = f"{self._base_url}/{self._index}/_search"
            resp = await self.client.post(search_url, json=body)
            resp.raise_for_status()
            data = resp.json()

            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                return SourceDocument(
                    url=url, title="", content="",
                    source_type=SourceType.ELASTIC,
                    error=f"Not found in the Elastic index: {url}",
                )

            source = hits[0].get("_source", {})
            title = source.get(self.config.field_title, "")
            body_text = source.get(self.config.field_body, "")

            return SourceDocument(
                url=url,
                title=title,
                content=body_text,
                source_type=SourceType.ELASTIC,
                content_length=len(body_text),
            )

        except Exception as e:
            logger.warning(f"Elasticsearch fetch failed for {url}: {e}")
            return SourceDocument(
                url=url, title="", content="",
                source_type=SourceType.ELASTIC,
                error=str(e),
            )

    def can_handle(self, url: str) -> bool:
        """Check whether the URL belongs to the configured website."""
        if not self._site_base_url:
            return False
        return url.lower().startswith(self._site_base_url.lower())

    async def close(self):
        await self.client.aclose()
