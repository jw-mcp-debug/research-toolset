"""
URL safety validation against SSRF (server-side request forgery).

Prevents user-supplied URLs from pointing at internal endpoints:
  - localhost / 127.0.0.1 / ::1, including short and numeric IPv4 forms
  - private IP ranges (RFC 1918): 10/8, 172.16/12, 192.168/16
  - link-local (169.254/16), e.g. a cloud metadata service
  - any other address that is not globally routable
  - file://, ftp://, gopher:// and other non-HTTP schemes
  - URLs without a host

Call sites:
  - `WebScraperConnector.fetch(url)` — direct URL fetches (lexical check)
  - `make_request_guard()` — httpx hook on the scraper and GitHub raw
    clients; resolves the host and checks every request and redirect hop
  - `validate_plan_conventions` (for direct_urls in the plan)

Configurable: installations that must allow internal endpoints (e.g. an
internal wiki in 10.0.0.0/8) can use the `allowed_internal_hosts` allow
list (FETCH_ALLOWED_INTERNAL_HOSTS).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# Allowed URL schemes — all others (file, ftp, gopher, javascript, ...)
# are rejected
ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafeURLError(ValueError):
    """Raised when a URL must not be fetched."""

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"URL not allowed: {url!r} — {reason}")


def is_safe_public_url(
    url: str,
    allowed_internal_hosts: Optional[set[str]] = None,
) -> tuple[bool, str]:
    """Check whether a URL may safely be fetched as a public resource.

    Args:
        url: The URL to check.
        allowed_internal_hosts: Set of host names that are allowed despite
                                an internal IP address (e.g. an internal
                                wiki). Default: empty.

    Returns:
        (ok, reason). With `ok=True` reason is empty; with `ok=False`
        it describes why the URL was rejected.
    """
    if not url or not isinstance(url, str):
        return False, "URL is empty or not a string"

    url_stripped = url.strip()
    if not url_stripped:
        return False, "URL is empty"

    try:
        parsed = urlparse(url_stripped)
    except Exception as e:
        return False, f"parse error: {e}"

    # 1. Scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return False, f"scheme {scheme!r} not allowed (only http/https)"

    # 2. Host must be present
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "no host in the URL"

    # 3. Allow list?
    if allowed_internal_hosts and host in allowed_internal_hosts:
        return True, ""

    # 4. Localhost aliases
    if host in ("localhost", "localhost.localdomain", "ip6-localhost"):
        return False, "localhost is not allowed"

    # 5. IP literal? Also accepts the short and numeric forms that the
    # resolver understands (127.1, 0x7f000001, 2130706433); a purely
    # textual check would let them through.
    ip = _parse_ip_literal(host)
    if ip is None:
        # Hostname: the lexical check cannot decide. `assert_safe_resolved`
        # (used before every outgoing request) resolves it and checks
        # every resulting address.
        return True, ""

    return _classify_ip(ip)
    return True, ""


def _parse_ip_literal(host: str):
    """Return an ip_address for IP literals in any resolver-accepted form."""
    candidate = host.strip("[]")
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        pass
    # IPv4 shorthand / hex / octal / single integer forms. inet_aton
    # accepts exactly what the C resolver would connect to.
    if candidate and all(c in "0123456789abcdefxABCDEFX." for c in candidate):
        try:
            return ipaddress.ip_address(socket.inet_aton(candidate))
        except OSError:
            return None
    return None


def _classify_ip(ip) -> tuple[bool, str]:
    """Reject every address that is not globally routable."""
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    if ip.is_loopback:
        return False, f"loopback address {ip} is not allowed"
    if ip.is_private:
        return False, f"private IP {ip} is not allowed"
    if ip.is_link_local:
        return False, (
            f"link-local address {ip} is not allowed "
            f"(e.g. cloud metadata service)"
        )
    if ip.is_multicast:
        return False, f"multicast address {ip} is not allowed"
    if ip.is_reserved:
        return False, f"reserved address {ip} is not allowed"
    if ip.is_unspecified:
        return False, f"unspecified address {ip}"
    if not ip.is_global:
        # e.g. 100.64.0.0/10 (shared address space), which is neither
        # private nor public in the ipaddress module's sense.
        return False, f"address {ip} is not publicly routable"
    return True, ""


async def assert_safe_resolved(
    url: str,
    allowed_internal_hosts: Optional[set[str]] = None,
) -> None:
    """Lexical check plus DNS resolution of the host.

    Raises UnsafeURLError if the URL itself or ANY address the host
    resolves to is not publicly routable. Hosts on the allow list are
    accepted without resolution.

    Residual risk: the connection resolves again (no address pinning),
    so a DNS rebinding between check and connect is not prevented.
    """
    assert_safe_url(url, allowed_internal_hosts)
    host = (urlparse(url.strip()).hostname or "").lower()
    if allowed_internal_hosts and host in allowed_internal_hosts:
        return
    if _parse_ip_literal(host) is not None:
        return  # already classified by assert_safe_url
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise UnsafeURLError(url, f"host cannot be resolved: {e}") from e
    for info in infos:
        ok, reason = _classify_ip(ipaddress.ip_address(info[4][0]))
        if not ok:
            raise UnsafeURLError(url, f"{host} → {reason}")


def make_request_guard(allowed_internal_hosts: Optional[set[str]] = None):
    """httpx event hook that checks EVERY outgoing request.

    Registered as `event_hooks={"request": [hook]}`, it also runs for
    each redirect hop, so a public URL cannot redirect to an internal
    address.
    """
    async def _guard(request) -> None:
        await assert_safe_resolved(str(request.url), allowed_internal_hosts)
    return _guard


def parse_allowed_hosts(raw: str) -> set[str]:
    """Comma-separated host list from configuration → lower-case set."""
    return {h.strip().lower() for h in (raw or "").split(",") if h.strip()}


def assert_safe_url(
    url: str,
    allowed_internal_hosts: Optional[set[str]] = None,
) -> None:
    """Raise UnsafeURLError if the URL must not be fetched.

    Convenience wrapper around `is_safe_public_url`.
    """
    ok, reason = is_safe_public_url(url, allowed_internal_hosts)
    if not ok:
        raise UnsafeURLError(url, reason)
