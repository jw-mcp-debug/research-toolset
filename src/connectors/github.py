"""
GitHub REST API connector — free of charge, 5000 requests/h with a personal access token.
"""

import asyncio
import base64
import logging
import re
import time

import httpx

from src.connectors.base import BaseConnector
from src.pipeline.models import SearchResult, SourceDocument, SourceType
from src.config import GitHubConfig

logger = logging.getLogger(__name__)


class GitHubConnector(BaseConnector):
    """GitHub REST API v3 connector."""

    name = "github"

    def __init__(self, config: GitHubConfig):
        self.config = config
        headers = {"Accept": "application/vnd.github.v3+json"}
        if config.token:
            headers["Authorization"] = f"token {config.token}"
            headers["Accept"] = "application/vnd.github.v3.text-match+json"

        self.client = httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        )
        # Separate client for arbitrary URLs in `_raw_fetch`: it must never
        # carry the API token, and every request (including redirects) is
        # checked against internal addresses.
        from src.core.url_security import make_request_guard
        self._raw_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            event_hooks={"request": [make_request_guard()]},
        )

        # Rate limiting: GitHub Search API = 10 req/min (unauthenticated) / 30 req/min (authenticated)
        self._has_token = bool(config.token)
        max_concurrent = 5 if self._has_token else 2
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limited = False
        self._rate_limit_reset = 0.0
        self._request_count = 0
        self._error_count = 0

        if not self._has_token:
            logger.warning(
                "⚠️ GitHub: no token configured. Rate limit: 10 searches/min, "
                "60 req/h. Set GITHUB_TOKEN in .env for 5000 req/h."
            )

    async def _rate_limited_request(self, method: str, url: str,
                                     **kwargs) -> httpx.Response | None:
        """Run an API request with rate-limit handling."""
        async with self._semaphore:
            # Wait while rate-limited
            if self._rate_limited:
                wait_until = self._rate_limit_reset - time.time()
                if wait_until > 0:
                    if wait_until > 60:
                        logger.info(f"GitHub: rate limit, skipping ({wait_until:.0f}s)")
                        return None
                    logger.info(f"GitHub: rate limit, waiting {wait_until:.0f}s...")
                    await asyncio.sleep(min(wait_until + 1, 30))
                self._rate_limited = False

            # Short pause between requests (to stay clear of the rate limit)
            if self._request_count > 0:
                delay = 0.2 if self._has_token else 1.0
                await asyncio.sleep(delay)

            self._request_count += 1

            try:
                resp = await self.client.request(method, url, **kwargs)

                # Evaluate the rate-limit headers
                remaining = resp.headers.get("x-ratelimit-remaining", "")
                reset = resp.headers.get("x-ratelimit-reset", "")

                if resp.status_code == 403 and "rate limit" in resp.text.lower():
                    self._rate_limited = True
                    try:
                        self._rate_limit_reset = float(reset) if reset else time.time() + 60
                    except ValueError:
                        self._rate_limit_reset = time.time() + 60
                    logger.warning(
                        f"GitHub: rate limit reached! "
                        f"Reset in {self._rate_limit_reset - time.time():.0f}s. "
                        f"(requests so far: {self._request_count})"
                    )
                    return None

                if remaining and int(remaining) < 5:
                    logger.info(f"GitHub: only {remaining} requests left")

                self._error_count = 0
                return resp

            except Exception as e:
                self._error_count += 1
                if self._error_count <= 3:
                    logger.warning(f"GitHub request failed: {e}")
                return None

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search GitHub: repositories + code + issues."""
        if self._rate_limited:
            return []

        results = []

        # 1. Repository search
        resp = await self._rate_limited_request(
            "GET", "/search/repositories",
            params={"q": query, "per_page": min(max_results, 5),
                    "sort": "stars", "order": "desc"},
        )
        if resp and resp.status_code == 200:
            for repo in resp.json().get("items", []):
                results.append(SearchResult(
                    title=repo["full_name"],
                    url=repo["html_url"],
                    snippet=repo.get("description", "") or "",
                    source_type=SourceType.GIT_REPO,
                    connector_name=self.name,
                ))

        # 2. Code search (only with a token, and only when not rate-limited)
        if self._has_token and not self._rate_limited:
            resp = await self._rate_limited_request(
                "GET", "/search/code",
                params={"q": query, "per_page": min(max_results, 5)},
            )
            if resp and resp.status_code == 200:
                for item in resp.json().get("items", []):
                    snippet = ""
                    text_matches = item.get("text_matches", [])
                    if text_matches:
                        snippet = text_matches[0].get("fragment", "")
                    results.append(SearchResult(
                        title=f"{item['repository']['full_name']}/{item['path']}",
                        url=item["html_url"],
                        snippet=snippet,
                        source_type=SourceType.GIT_FILE,
                        connector_name=self.name,
                    ))

        logger.info(f"GitHub: '{query}' → {len(results)} results")
        return results[:max_results]

    async def fetch(self, url: str) -> SourceDocument:
        """Fetch a file or README via the GitHub API."""
        t0 = time.monotonic()

        parts = self._parse_github_url(url)
        if not parts:
            # Fallback: Raw-Fetch
            return await self._raw_fetch(url, t0)

        owner, repo = parts["owner"], parts["repo"]
        path = parts.get("path", "")

        try:
            if not path:
                # Load the README
                return await self._fetch_readme(owner, repo, url, t0)
            else:
                # Load the file
                return await self._fetch_file(owner, repo, path, url, t0)
        except Exception as e:
            logger.warning(f"GitHub API fetch failed for {url}: {e}")
            return await self._raw_fetch(url, t0)

    async def search_org_repos(self, org: str,
                               max_results: int = 100) -> list[SearchResult]:
        """List the repositories of an organisation."""
        results = []
        page = 1
        while len(results) < max_results:
            if self._rate_limited:
                break
            resp = await self._rate_limited_request(
                "GET", f"/orgs/{org}/repos",
                params={"per_page": 100, "page": page, "sort": "updated"},
            )
            if not resp or resp.status_code != 200:
                break
            repos = resp.json()
            if not repos:
                break
            for repo in repos:
                results.append(SearchResult(
                    title=repo["full_name"],
                    url=repo["html_url"],
                    snippet=repo.get("description", "") or "",
                    source_type=SourceType.GIT_REPO,
                    connector_name=self.name,
                ))
            page += 1
        return results[:max_results]

    async def search_issues(self, repo_fullname: str, query: str,
                            max_results: int = 10) -> list[SearchResult]:
        """Search the issues + PRs of a repository."""
        if self._rate_limited:
            return []

        resp = await self._rate_limited_request(
            "GET", "/search/issues",
            params={"q": f"{query} repo:{repo_fullname}",
                    "per_page": max_results, "sort": "updated"},
        )
        if not resp or resp.status_code != 200:
            return []

        results = []
        for item in resp.json().get("items", []):
            kind = "PR" if "pull_request" in item else "Issue"
            results.append(SearchResult(
                title=f"{kind} #{item['number']}: {item['title']}",
                url=item["html_url"],
                snippet=(item.get("body", "") or "")[:300],
                source_type=SourceType.GIT_ISSUE,
                connector_name=self.name,
            ))
        return results

    def can_handle(self, url: str) -> bool:
        return "github.com" in url or "raw.githubusercontent.com" in url

    async def close(self):
        await self.client.aclose()
        await self._raw_client.aclose()

    # ─── Helper functions ────────────────────────────────────────────

    async def _fetch_readme(self, owner: str, repo: str,
                            url: str, t0: float) -> SourceDocument:
        resp = await self.client.get(f"/repos/{owner}/{repo}/readme")
        resp.raise_for_status()
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return SourceDocument(
            source_type=SourceType.GIT_FILE,
            url=url,
            title=f"{owner}/{repo}/README",
            content=content,
            fetch_time_seconds=time.monotonic() - t0,
        )

    async def _fetch_file(self, owner: str, repo: str, path: str,
                          url: str, t0: float) -> SourceDocument:
        resp = await self.client.get(f"/repos/{owner}/{repo}/contents/{path}")
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            # directory → file list
            content = f"# Directory: {owner}/{repo}/{path}\n\n"
            for item in data:
                content += f"- {item['name']} ({item['type']}, {item.get('size', '?')} bytes)\n"
        elif data.get("encoding") == "base64":
            content = base64.b64decode(data["content"]).decode("utf-8")
        else:
            content = data.get("content", "")

        return SourceDocument(
            source_type=SourceType.GIT_FILE,
            url=url,
            title=f"{owner}/{repo}/{path}",
            content=content,
            fetch_time_seconds=time.monotonic() - t0,
        )

    async def _raw_fetch(self, url: str, t0: float) -> SourceDocument:
        """Fallback: fetch directly.

        This path accepts URLs that were not recognised as regular
        github.com URLs, so they can point anywhere. It therefore uses a
        separate client without the API token, and every request
        (including redirects) passes the SSRF guard.
        """
        from src.core.url_security import UnsafeURLError, assert_safe_url
        try:
            assert_safe_url(url)
        except UnsafeURLError as e:
            logger.warning("URL rejected (SSRF protection): %s — %s", url, e.reason)
            return SourceDocument(
                source_type=SourceType.WEB_PAGE,
                url=url, title=url,
                content=f"[URL rejected: {e.reason}]",
                metadata={"fetch_error": f"URL rejected: {e.reason}"},
                fetch_time_seconds=time.monotonic() - t0,
            )
        try:
            resp = await self._raw_client.get(url)
            content = resp.text
            return SourceDocument(
                source_type=SourceType.WEB_PAGE,
                url=url, title=url,
                content=content[:100000],
                fetch_time_seconds=time.monotonic() - t0,
            )
        except Exception as e:
            return SourceDocument(
                source_type=SourceType.WEB_PAGE,
                url=url, title=url,
                content=f"[Error: {e}]",
                fetch_time_seconds=time.monotonic() - t0,
            )

    @staticmethod
    def _parse_github_url(url: str) -> dict | None:
        """Parse github.com/{owner}/{repo}[/blob/{branch}/{path}]"""
        patterns = [
            # github.com/owner/repo/blob/branch/path
            r"github\.com/([^/]+)/([^/]+)/blob/[^/]+/(.+)",
            # github.com/owner/repo/tree/branch/path
            r"github\.com/([^/]+)/([^/]+)/tree/[^/]+/(.+)",
            # github.com/owner/repo
            r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
            # raw.githubusercontent.com/owner/repo/branch/path
            r"raw\.githubusercontent\.com/([^/]+)/([^/]+)/[^/]+/(.+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, url)
            if m:
                groups = m.groups()
                result = {"owner": groups[0], "repo": groups[1]}
                if len(groups) > 2:
                    result["path"] = groups[2]
                return result
        return None
