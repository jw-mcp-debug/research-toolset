"""
GitLab REST API connector — for self-hosted GitLab instances.
"""

import base64
import logging
import time
from urllib.parse import quote_plus

import httpx

from src.connectors.base import BaseConnector
from src.pipeline.models import SearchResult, SourceDocument, SourceType
from src.config import GitLabConfig

logger = logging.getLogger(__name__)


class GitLabConnector(BaseConnector):
    """GitLab REST API v4 connector."""

    name = "gitlab"

    def __init__(self, config: GitLabConfig):
        self.config = config
        headers = {}
        if config.token:
            headers["PRIVATE-TOKEN"] = config.token

        self.client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        )

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search GitLab: projects + code."""
        results = []

        # Project search
        try:
            resp = await self.client.get("/api/v4/projects", params={
                "search": query, "per_page": min(max_results, 10),
                "order_by": "last_activity_at",
            })
            if resp.status_code == 200:
                for proj in resp.json():
                    results.append(SearchResult(
                        title=proj.get("path_with_namespace", ""),
                        url=proj.get("web_url", ""),
                        snippet=proj.get("description", "") or "",
                        source_type=SourceType.GIT_REPO,
                        connector_name=self.name,
                    ))
        except Exception as e:
            logger.warning(f"GitLab project search failed: {e}")

        # Code search (blobs)
        try:
            resp = await self.client.get("/api/v4/search", params={
                "scope": "blobs", "search": query, "per_page": min(max_results, 5),
            })
            if resp.status_code == 200:
                for item in resp.json():
                    proj_id = item.get("project_id", "")
                    path = item.get("path", "")
                    results.append(SearchResult(
                        title=f"{item.get('project_name', proj_id)}/{path}",
                        url=f"{self.config.base_url}/projects/{proj_id}/-/blob/main/{path}",
                        snippet=item.get("data", "")[:300],
                        source_type=SourceType.GIT_FILE,
                        connector_name=self.name,
                    ))
        except Exception as e:
            logger.warning(f"GitLab code search failed: {e}")

        logger.info(f"GitLab: '{query}' → {len(results)} results")
        return results[:max_results]

    async def fetch(self, url: str) -> SourceDocument:
        """Fetch a file via the GitLab API."""
        t0 = time.monotonic()

        parts = self._parse_gitlab_url(url)
        if not parts:
            return SourceDocument(
                source_type=SourceType.GIT_FILE,
                url=url, title=url,
                content=f"[GitLab URL not parsable: {url}]",
                fetch_time_seconds=time.monotonic() - t0,
            )

        project_path = parts["project"]
        file_path = parts.get("path", "")
        encoded_project = quote_plus(project_path)

        try:
            if not file_path:
                # Load the README
                resp = await self.client.get(
                    f"/api/v4/projects/{encoded_project}/repository/files/README.md",
                    params={"ref": "main"},
                )
                if resp.status_code != 200:
                    resp = await self.client.get(
                        f"/api/v4/projects/{encoded_project}/repository/files/README.md",
                        params={"ref": "master"},
                    )
            else:
                encoded_path = quote_plus(file_path)
                resp = await self.client.get(
                    f"/api/v4/projects/{encoded_project}/repository/files/{encoded_path}",
                    params={"ref": "main"},
                )
                if resp.status_code != 200:
                    resp = await self.client.get(
                        f"/api/v4/projects/{encoded_project}/repository/files/{encoded_path}",
                        params={"ref": "master"},
                    )

            if resp.status_code == 200:
                data = resp.json()
                content = base64.b64decode(data["content"]).decode("utf-8")
                return SourceDocument(
                    source_type=SourceType.GIT_FILE,
                    url=url,
                    title=f"{project_path}/{file_path or 'README.md'}",
                    content=content,
                    fetch_time_seconds=time.monotonic() - t0,
                )
            else:
                return SourceDocument(
                    source_type=SourceType.GIT_FILE,
                    url=url, title=url,
                    content=f"[GitLab HTTP {resp.status_code}: {url}]",
                    fetch_time_seconds=time.monotonic() - t0,
                )

        except Exception as e:
            logger.error(f"GitLab fetch failed: {e}")
            return SourceDocument(
                source_type=SourceType.GIT_FILE,
                url=url, title=url,
                content=f"[GitLab error: {e}]",
                fetch_time_seconds=time.monotonic() - t0,
            )

    def can_handle(self, url: str) -> bool:
        if not self.config.base_url:
            return False
        from urllib.parse import urlparse
        gl_host = urlparse(self.config.base_url).hostname or ""
        return gl_host in url

    async def close(self):
        await self.client.aclose()

    def _parse_gitlab_url(self, url: str) -> dict | None:
        """Parse a GitLab URL into project + path."""
        import re
        from urllib.parse import urlparse

        gl_host = urlparse(self.config.base_url).hostname or ""
        # gitlab.example.com/group/project/-/blob/branch/path
        m = re.search(
            rf"{re.escape(gl_host)}/(.+?)/-/blob/[^/]+/(.+)", url
        )
        if m:
            return {"project": m.group(1), "path": m.group(2)}

        # gitlab.example.com/group/project
        m = re.search(rf"{re.escape(gl_host)}/(.+?)/?$", url)
        if m:
            return {"project": m.group(1)}

        return None
