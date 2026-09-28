"""
Web scraper connector: fetches web pages and extracts text.
Primary: httpx + trafilatura
Fallback: playwright (optional)
"""

import asyncio
import logging
import time
from collections import defaultdict

import httpx

from src.connectors.base import BaseConnector
from src.pipeline.models import SearchResult, SourceDocument, SourceType
from src.config import WebScraperConfig

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _normalize_iso_date(s: str) -> str:
    """Normalise a date string to YYYY-MM-DD.

    Accepts ISO 8601 variants (with/without time zone, with/without a
    time part) and some common formats from meta tags. For unparsable
    or obviously invalid strings an empty string is returned — never a
    guessed date.
    """
    import re
    if not s:
        return ""
    s = s.strip()

    # ISO with a time part: 2016-09-08T10:30:00+02:00 → 2016-09-08
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = m.groups()
        if 1900 <= int(y) <= 2100 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"

    # German format DD.MM.YYYY → YYYY-MM-DD
    m = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
    if m:
        d, mo, y = m.groups()
        if 1900 <= int(y) <= 2100 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    # The US format MM/DD/YYYY cannot be told apart from DD/MM/YYYY —
    # deliberately NOT accepted. Better an empty string than a wrong reading.

    return ""


class WebScraperConnector(BaseConnector):
    """Fetch web pages and extract structured text."""

    name = "web_scraper"

    def __init__(self, config: WebScraperConfig):
        self.config = config
        from src.core.url_security import make_request_guard, parse_allowed_hosts
        self._allowed_hosts = parse_allowed_hosts(
            getattr(config, "allowed_internal_hosts", "")
        )
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            # Checks every request, including each redirect hop and the
            # raw-variant fetches below, after DNS resolution.
            event_hooks={"request": [make_request_guard(self._allowed_hosts)]},
        )
        # Per-Domain Rate-Limiting (A4)
        self._domain_last_request: dict[str, float] = defaultdict(float)
        # ONE lock PER domain — not one for all. The sleep below runs while
        # the lock is held; a single shared lock would make a fetch that
        # waits 1 s on one host block every fetch on every other domain too,
        # rendering MAX_PARALLEL_FETCHES practically ineffective.
        # The guarantee "at least `_domain_delay` between two requests to the
        # same host" is unchanged.
        self._domain_locks: dict[str, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )
        self._domain_delay = 1.0  # at least 1 s between requests to the same domain
        # Fetch-Cache (D1): URL → (SourceDocument, timestamp)
        self._cache: dict[str, tuple[SourceDocument, float]] = {}
        self._cache_ttl = 3600.0  # 1 hour default TTL

    async def search(self, query: str, max_results: int = 10, **kwargs
                     ) -> list[SearchResult]:
        """The web scraper does not search — fetch only."""
        return []

    async def _wait_for_domain(self, url: str):
        """Per-Domain Rate-Limiting."""
        from urllib.parse import urlparse
        domain = (urlparse(url).hostname or "").lower()
        if not domain:
            return
        async with self._domain_locks[domain]:
            now = time.monotonic()
            last = self._domain_last_request[domain]
            wait = self._domain_delay - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._domain_last_request[domain] = time.monotonic()

    async def fetch(self, url: str) -> SourceDocument:
        """Fetch a URL and extract its text (with cache).

        Security (SSRF): the URL is checked lexically against internal
        endpoints (localhost, RFC 1918 private addresses, link-local such
        as a cloud metadata service, file:// etc.); in addition, the
        client's request hook resolves the host and checks every request
        and redirect hop. A rejection is turned into a `SourceDocument`
        stub with an error marker, so that a single bad URL in the plan
        does not bring down the whole pipeline.
        """
        from src.core.url_security import (
            UnsafeURLError, assert_safe_url,
        )
        try:
            assert_safe_url(url, self._allowed_hosts)
        except UnsafeURLError as e:
            logger.warning("URL rejected (SSRF protection): %s — %s", url, e.reason)
            return SourceDocument(
                source_type=SourceType.WEB_SEARCH,
                url=url, title="", content="",
                metadata={"fetch_error": f"URL rejected: {e.reason}"},
                fetch_time_seconds=0.0,
            )

        # Check the cache
        if url in self._cache:
            doc, cached_at = self._cache[url]
            if time.monotonic() - cached_at < self._cache_ttl:
                logger.debug(f"Cache hit: {url}")
                return doc

        t0 = time.monotonic()
        await self._wait_for_domain(url)

        try:
            resp = await self.client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            raw_bytes = resp.content

            # ── PDF detection (Content-Type OR %PDF magic) ──────────
            # resp.text on PDF bytes yields decoded binary garbage; trafilatura
            # discards it ("discarding data") and the simple-strip fallback
            # returns ~500k characters of binary junk. That would end up in the
            # reranker (snippet ranking), in the off-topic classifier
            # (content[:500] = PDF header → discarded as off_topic) and in the
            # harvest, and relevant papers (arXiv PDFs!) would be lost. Instead:
            # extract the real text with pdfminer, and on error fall back CLEANLY
            # to a short marker (NOT to binary junk).
            is_pdf = (
                "application/pdf" in content_type.lower()
                or raw_bytes[:5] == b"%PDF-"
            )
            if is_pdf:
                content = await self._extract_pdf_text(raw_bytes, url)
                elapsed = time.monotonic() - t0
                title = url.rsplit("/", 1)[-1] or url
                if len(content) > self.config.max_content_length:
                    content = (
                        content[:self.config.max_content_length]
                        + "\n\n[... shortened]"
                    )
                logger.info(
                    "WebScraper(PDF): %s → %d characters in %.1fs",
                    url, len(content), elapsed,
                )
                doc = SourceDocument(
                    source_type=SourceType.WEB_PAGE,
                    url=url, title=title, content=content,
                    fetch_time_seconds=elapsed,
                    metadata={"content_type": "application/pdf"},
                )
                self._cache[url] = (doc, time.monotonic())
                return doc

            raw_html = resp.text

            # Extract text with trafilatura
            content = await self._extract_with_trafilatura(raw_html, url)

            # Fallback: simple HTML stripping
            if not content or len(content) < self.config.playwright_min_content_length:
                content_simple = self._simple_html_strip(raw_html)
                if len(content_simple) > len(content or ""):
                    content = content_simple

            # Fallback: raw Markdown variant for known platforms.
            # Hugging Face, GitHub and GitLab serve Markdown files directly —
            # without JavaScript rendering, with the full content. Much faster
            # and more reliable than Playwright.
            if (not content or
                    len(content) < self.config.playwright_min_content_length):
                raw_url = self._try_raw_variant(url)
                if raw_url and raw_url != url:
                    try:
                        raw_resp = await self.client.get(raw_url)
                        if raw_resp.status_code == 200:
                            raw_text = raw_resp.text
                            if len(raw_text) > len(content or ""):
                                content = raw_text
                                logger.info(
                                    f"Raw variant used: {raw_url} "
                                    f"→ {len(raw_text)} characters"
                                )
                    except Exception as e:
                        logger.debug(
                            f"Raw variant failed for {raw_url}: {e}"
                        )

            # Optional: Playwright fallback for JS pages
            if (self.config.use_playwright_fallback and
                    (not content or len(content) < self.config.playwright_min_content_length)):
                try:
                    content_pw = await self._fetch_with_playwright(url)
                    if content_pw and len(content_pw) > len(content or ""):
                        content = content_pw
                except Exception as e:
                    logger.debug(f"Playwright fallback failed for {url}: {e}")

            if not content:
                content = f"[No text extractable from {url}]"

            if len(content) > self.config.max_content_length:
                content = content[:self.config.max_content_length] + "\n\n[... shortened]"

            elapsed = time.monotonic() - t0
            title = self._extract_title(raw_html) or url
            published_date = self._extract_published_date(raw_html)
            logger.info(
                f"WebScraper: {url} → {len(content)} characters in {elapsed:.1f}s"
                + (f" (publ. {published_date})" if published_date else "")
            )

            doc = SourceDocument(
                source_type=SourceType.WEB_PAGE,
                url=url, title=title, content=content,
                fetch_time_seconds=elapsed,
                published_date=published_date,
                metadata={"content_type": content_type},
            )
            # Store in the cache
            self._cache[url] = (doc, time.monotonic())
            return doc

        except UnsafeURLError as e:
            logger.warning("URL rejected (SSRF protection): %s — %s", url, e.reason)
            return SourceDocument(
                source_type=SourceType.WEB_SEARCH,
                url=url, title="", content="",
                metadata={"fetch_error": f"URL rejected: {e.reason}"},
                fetch_time_seconds=time.monotonic() - t0,
            )
        except httpx.TimeoutException:
            logger.warning(f"WebScraper Timeout: {url}")
            return SourceDocument(
                source_type=SourceType.WEB_PAGE, url=url, title=url,
                content=f"[Timeout while loading {url}]",
                fetch_time_seconds=time.monotonic() - t0,
            )
        except httpx.HTTPStatusError as e:
            logger.warning(f"WebScraper HTTP {e.response.status_code}: {url}")
            return SourceDocument(
                source_type=SourceType.WEB_PAGE, url=url, title=url,
                content=f"[HTTP error {e.response.status_code} for {url}]",
                fetch_time_seconds=time.monotonic() - t0,
            )
        except Exception as e:
            logger.error(f"WebScraper error: {url}: {e}")
            return SourceDocument(
                source_type=SourceType.WEB_PAGE, url=url, title=url,
                content=f"[Error while loading {url}: {e}]",
                fetch_time_seconds=time.monotonic() - t0,
            )

    def can_handle(self, url: str) -> bool:
        return url.startswith("http://") or url.startswith("https://")

    async def close(self):
        await self.client.aclose()

    # ─── Helper functions ────────────────────────────────────────

    @staticmethod
    async def _extract_pdf_text(raw_bytes: bytes, url: str) -> str:
        """Extract text from PDF bytes via pdfminer.six.

        pdfminer is a declared dependency anyway and is already used in
        src/documents/processor.py — the same proven path. FAIL-CLEAN: on
        any error a SHORT marker instead of binary junk, so that
        reranker / off-topic filter / harvest are not poisoned by 500k
        characters of raw PDF bytes.
        """
        try:
            import io
            from pdfminer.high_level import extract_text

            def _do() -> str:
                return extract_text(io.BytesIO(raw_bytes)) or ""

            text = await asyncio.to_thread(_do)
            text = (text or "").strip()
            if len(text) < 50:
                # Scanned / image-based PDF without a text layer:
                # NO OCR here — mark it cleanly instead of producing junk.
                return (
                    f"[PDF without an extractable text layer: {url} "
                    f"— possibly a scanned document]"
                )
            return text
        except Exception as e:
            logger.debug("PDF extraction failed for %s: %s", url, e)
            return f"[PDF not readable: {url} ({type(e).__name__})]"

    @staticmethod
    async def _extract_with_trafilatura(html: str, url: str) -> str:
        """Extract the main text with trafilatura."""
        try:
            import trafilatura
            result = await asyncio.to_thread(
                trafilatura.extract,
                html, url=url,
                include_comments=False,
                include_tables=True,
                include_links=True,   # keep links for link following
                output_format="txt",
                favor_precision=False,
                favor_recall=True,
            )
            return result or ""
        except ImportError:
            logger.warning("trafilatura not installed")
            return ""
        except Exception as e:
            logger.debug(f"trafilatura error: {e}")
            return ""

    @staticmethod
    def _try_raw_variant(url: str) -> str | None:
        """Map JavaScript-heavy platform URLs to their raw-text variants
        where possible.

        Known platforms such as Hugging Face, GitHub and GitLab serve
        Markdown content directly as raw files. These endpoints are much
        faster and more reliable than the JavaScript-rendered main UI —
        and the scraper gets the full content instead of just the frame.

        Returns:
            An alternative URL with the same content as raw text,
            or None if no raw variant is known.
        """
        if not url:
            return None

        # Hugging Face: /model/repo → /model/repo/raw/main/README.md
        # Works for models, datasets and spaces.
        if "huggingface.co/" in url:
            # Already a resolve/raw URL? Then do not map further.
            if "/resolve/" in url or "/raw/" in url or "/blob/" in url:
                return None
            # Remove the trailing slash
            clean = url.rstrip("/")
            # Only model/dataset URLs (two path segments after the host),
            # not search pages or profiles.
            parts = clean.replace("https://huggingface.co/", "").split("/")
            if len(parts) >= 2 and all(p for p in parts[:2]):
                return f"{clean}/raw/main/README.md"

        # GitHub: blob/main/... → raw.githubusercontent.com/.../main/...
        # For README pages and other Markdown files in the repository.
        if "github.com/" in url and "/blob/" in url:
            return url.replace(
                "https://github.com/", "https://raw.githubusercontent.com/"
            ).replace("/blob/", "/", 1)

        # GitHub: repository root → raw README.md (main or master)
        # Only if there are no further path segments after owner/repo.
        if "github.com/" in url:
            clean = url.rstrip("/")
            parts = clean.replace("https://github.com/", "").split("/")
            # exactly owner/repo without further segments
            if len(parts) == 2 and all(p for p in parts):
                owner, repo = parts
                return (
                    f"https://raw.githubusercontent.com/"
                    f"{owner}/{repo}/main/README.md"
                )

        # GitLab: blob/main/... → raw/main/...
        if "gitlab.com/" in url and "/-/blob/" in url:
            return url.replace("/-/blob/", "/-/raw/")

        return None

    @staticmethod
    def _simple_html_strip(html: str) -> str:
        """Simple HTML → text as a fallback."""
        import re
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html,
                       flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        try:
            import html as html_lib
            text = html_lib.unescape(text)
        except Exception:
            pass
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_title(html: str) -> str:
        """Extract <title> from HTML."""
        import re
        match = re.search(r"<title[^>]*>(.*?)</title>", html,
                          re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            try:
                import html as html_lib
                title = html_lib.unescape(title)
            except Exception:
                pass
            return title[:200]
        return ""

    @staticmethod
    def _extract_published_date(html: str) -> str:
        """Extract the publication date from HTML metadata.

        In order of reliability:
        1. <meta property="article:published_time"> (Open Graph)
        2. <meta name="date|dc.date|dcterms.issued|publish-date">
        3. JSON-LD: "datePublished": "..."
        4. <time datetime="..."> (first occurrence)

        Returns an ISO 8601 string (e.g. "2016-09-08") or an empty string.
        A deep search for dates in running text is deliberately NOT done —
        too error-prone (footer copyright, event dates, quotations). If the
        source has no structured date, an empty string is more honest
        than a guessed date.
        """
        import re
        if not html:
            return ""

        # Pattern 1: Open Graph article:published_time
        m = re.search(
            r'<meta\s+property=["\']article:published_time["\']\s+'
            r'content=["\']([^"\']+)["\']',
            html, re.IGNORECASE,
        )
        if m:
            return _normalize_iso_date(m.group(1))

        # Pattern 2: various meta name variants
        for name in ("date", "dc.date", "dcterms.issued", "dcterms.created",
                     "publish-date", "publication_date", "pubdate"):
            m = re.search(
                rf'<meta\s+name=["\']{re.escape(name)}["\']\s+'
                r'content=["\']([^"\']+)["\']',
                html, re.IGNORECASE,
            )
            if m:
                return _normalize_iso_date(m.group(1))

        # Pattern 3: JSON-LD datePublished
        # Restrict the scope to LD script tags, so that we do not
        # accidentally take datePublished from an embedded comment or
        # advertising script.
        ld_scripts = re.findall(
            r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, re.IGNORECASE | re.DOTALL,
        )
        for script in ld_scripts:
            m = re.search(
                r'"datePublished"\s*:\s*"([^"]+)"',
                script,
            )
            if m:
                return _normalize_iso_date(m.group(1))

        # Pattern 4: <time datetime="...">
        m = re.search(
            r'<time[^>]+datetime=["\']([^"\']+)["\']',
            html, re.IGNORECASE,
        )
        if m:
            return _normalize_iso_date(m.group(1))

        return ""

    @staticmethod
    async def _fetch_with_playwright(url: str) -> str:
        """Playwright fallback for JS-rendered pages."""
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)
                content = await page.content()
                await browser.close()
                import trafilatura
                return trafilatura.extract(content, url=url) or ""
        except Exception as e:
            logger.debug(f"Playwright error: {e}")
            return ""
