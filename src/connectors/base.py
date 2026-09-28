"""
Base interface for connectors and the central registry.
"""

import logging
import re
from abc import ABC, abstractmethod
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from src.pipeline.models import SearchResult, SourceDocument

logger = logging.getLogger(__name__)


def clean_url(url: str) -> str:
    """Strip common artefacts from URLs (Markdown brackets, quotes)."""
    url = url.strip()
    changed = True
    while changed:
        changed = False
        # trailing Markdown/JSON artefacts
        while url and url[-1] in "]}.,:;'\"":
            url = url[:-1]
            changed = True
        # Closing bracket only if there is no opening one in the last path segment
        if url.endswith(")") and "(" not in url.split("/")[-1]:
            url = url[:-1]
            changed = True
    return url


# ─── Domain block list (safety & quality) ─────────────────────

# Domains that should never appear in research results.
# Matched against the host name (without www.).
_BLOCKED_DOMAINS = {
    # Adult / NSFW
    "pornhub.com", "xvideos.com", "xnxx.com", "xhamster.com",
    "redtube.com", "youporn.com", "spankbang.com", "tube8.com",
    "brazzers.com", "onlyfans.com", "chaturbate.com", "livejasmin.com",
    "stripchat.com", "bongacams.com", "cam4.com", "myfreecams.com",
    "fapello.com", "rule34.xxx", "rule34.paheal.net", "e621.net",
    "nhentai.net", "hanime.tv", "hentaihaven.xxx",
    # Piracy / warez
    "thepiratebay.org", "1337x.to", "rarbg.to", "nyaa.si",
    "fitgirl-repacks.site",
    # Malware / spam / URL shorteners
    "bit.ly", "tinyurl.com", "adf.ly",
    # Content farms (extremely low quality)
    "quora.com",
    # Spam marketplaces / irrelevant results
    "alibaba.com", "aliexpress.com", "dhgate.com", "made-in-china.com",
    "globalsources.com", "tradekey.com", "indiamart.com",
    "ebay.com", "amazon.com", "amazon.de", "etsy.com",
    # Streaming / entertainment (no research value)
    "hulu.com", "netflix.com", "disneyplus.com", "crunchyroll.com",
    "spotify.com", "twitch.tv",
    # Esoteric / sectarian
    "eckankar.org",
    # Generic search engines / portals (duplicates of results)
    "google.com", "google.de", "google.it", "google.fr", "google.es",
    "google.co.uk", "google.co.jp", "bing.com", "yahoo.com",
    "baidu.com", "yandex.com", "duckduckgo.com",
    "google.co.in", "google.com.br", "google.ru", "google.ca",
    "images.google.com", "images.google.de", "images.google.it",
    # App stores (no text content)
    "apps.apple.com", "play.google.com",
    # Yumpu / Slideshare (mostly unreadable scans)
    "yumpu.com", "slideshare.net", "issuu.com",
    # Social media (rarely relevant for research)
    "facebook.com", "instagram.com", "tiktok.com", "pinterest.com",
    # First-name / baby-name sites (false positives in person searches)
    "baby-vornamen.de", "vorname.com", "beliebte-vornamen.de",
    "babycenter.de", "babycenter.com", "babyclub.de",
    "familienleben.ch", "eltern.de", "urbia.de",
    "wasbedeutet.info", "bedeutung-von-namen.de", "namensforschung.net",
    "behindthename.com", "nameberry.com", "babynamewizard.com",
    # Tourism / travel (false positives, e.g. place names that resemble first names)
    "visitmalta.com", "timeout.com", "tripadvisor.de", "tripadvisor.com",
    "booking.com", "expedia.de", "expedia.com", "hotels.com",
    "lonelyplanet.com", "holidaycheck.de",
    # Chinese Q&A sites / forums (rarely relevant)
    "zhihu.com", "baike.baidu.com", "tieba.baidu.com",
    "csdn.net", "jianshu.com", "douban.com",
    # French / generic forums
    "zestedesavoir.com", "commentcamarche.net",
    # Dictionaries / translation sites (no research value)
    "dict.cc", "leo.org", "linguee.de", "deepl.com",
    "wiktionary.org", "pons.com",
    # Worldatlas / generic geography sites
    "worldatlas.com", "nationsonline.org",
}

# Host-name substrings that indicate adult content
_BLOCKED_DOMAIN_PATTERNS = [
    "porn", "xxx", "sex", "hentai", "nude", "nsfw",
    "escort", "camgirl", "livecam", "adultfriend",
    # First-name sites (pattern for unknown domains)
    "vorname", "babyname", "namensbed",
]

# URL paths that indicate listing pages (content too broad for harvesting)
_BLOCKED_URL_PATTERNS = [
    # GitHub listing/collection pages
    r"github\.com/search\?",             # search results
    r"github\.com/trending",             # trending
    r"github\.com/explore",              # explore
    r"github\.com/topics/",              # topic lists
    r"github\.com/collections/",         # curated collections
    r"github\.com/orgs/[^/]+/repositories$",  # organisation repository overview
    r"github\.com/[^/]+\?tab=repositories",   # user repository list
    r"github\.com/[^/]+\?tab=stars",          # user stars
    # GitLab listing pages
    r"gitlab\.[^/]+/explore",
    # Other aggregator pages
    r"awesome-.*readme",                 # awesome-X lists
]

import re as _re
_COMPILED_URL_PATTERNS = [_re.compile(p) for p in _BLOCKED_URL_PATTERNS]


def is_url_blocked(url: str) -> bool:
    """Check whether a URL is on the block list."""
    try:
        host = urlparse(url).hostname or ""
        host = host.lower().removeprefix("www.")

        # Institution-specific additions (e.g. sites that share the
        # institution's abbreviation) come from the institution profile.
        from src.institution import get_profile
        blocked = _BLOCKED_DOMAINS | set(get_profile().blocked_domains)

        # Exact domain match
        if host in blocked:
            return True

        # Check parent domains (e.g. de.pornhub.com)
        parts = host.split(".")
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in blocked:
                return True

        # Pattern match in the host name
        for pattern in _BLOCKED_DOMAIN_PATTERNS:
            if pattern in host:
                return True

        # URL path patterns (listing pages)
        url_lower = url.lower()
        for compiled in _COMPILED_URL_PATTERNS:
            if compiled.search(url_lower):
                return True

        return False
    except Exception:
        return False


class BaseConnector(ABC):
    """Base interface for all connectors."""

    name: str = "base"

    @abstractmethod
    async def search(self, query: str, max_results: int = 10, **kwargs) -> list[SearchResult]:
        ...

    @abstractmethod
    async def fetch(self, url: str) -> SourceDocument:
        ...

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        ...

    async def close(self):
        """Release resources."""
        pass


def normalize_url(url: str) -> str:
    """Normalise URLs for de-duplication."""
    try:
        url = clean_url(url)
        parsed = urlparse(url)

        # Normalise the scheme
        scheme = parsed.scheme.lower() or "https"

        # Remove www.
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]

        # Port only if non-standard
        port = parsed.port
        if port in (80, 443, None):
            netloc = host
        else:
            netloc = f"{host}:{port}"

        # Remove the trailing slash
        path = parsed.path.rstrip("/")

        # Remove tracking parameters
        tracking_params = {
            "utm_source", "utm_medium", "utm_campaign", "utm_content",
            "utm_term", "ref", "source", "fbclid", "gclid", "mc_cid",
            "mc_eid", "s", "share",
        }
        params = {
            k: v for k, v in parse_qs(parsed.query).items()
            if k.lower() not in tracking_params
        }
        query = urlencode(params, doseq=True)

        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url.strip()


class ConnectorRegistry:
    """Central registry: maps URLs to the right connector."""

    def __init__(self):
        self._connectors: list[BaseConnector] = []
        self._url_patterns: list[tuple[re.Pattern, BaseConnector]] = []
        self._search_connector: BaseConnector | None = None  # SearXNG
        self._web_scraper: BaseConnector | None = None        # default fetcher
        self._seen_urls: set[str] = set()

    def register(self, connector: BaseConnector,
                 url_patterns: list[str] | None = None,
                 is_search_engine: bool = False,
                 is_web_scraper: bool = False):
        """Register a connector with optional URL patterns."""
        self._connectors.append(connector)

        if url_patterns:
            for pattern in url_patterns:
                self._url_patterns.append(
                    (re.compile(pattern, re.IGNORECASE), connector)
                )

        if is_search_engine:
            self._search_connector = connector
        if is_web_scraper:
            self._web_scraper = connector

    def get_search_connector(self) -> BaseConnector | None:
        return self._search_connector

    def get_web_scraper(self) -> BaseConnector | None:
        return self._web_scraper

    def route_url(self, url: str) -> BaseConnector | None:
        """Find the matching connector for a URL."""
        for pattern, connector in self._url_patterns:
            if pattern.search(url):
                return connector
        return self._web_scraper

    def get_connector_by_name(self, name: str) -> BaseConnector | None:
        for c in self._connectors:
            if c.name == name:
                return c
        return None

    def is_url_seen(self, url: str) -> bool:
        """Check whether a URL (normalised) has already been processed."""
        return normalize_url(url) in self._seen_urls

    def mark_url_seen(self, url: str):
        self._seen_urls.add(normalize_url(url))

    def reset_seen(self):
        self._seen_urls.clear()

    @property
    def connectors(self) -> list[BaseConnector]:
        return list(self._connectors)

    async def close_all(self):
        for c in self._connectors:
            try:
                await c.close()
            except Exception as e:
                logger.warning(f"Error closing {c.name}: {e}")
