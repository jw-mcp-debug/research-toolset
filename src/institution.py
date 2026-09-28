"""Institution profile: everything that ties the tool to one organisation.

The tool itself knows no institution. An optional TOML file, pointed to
by INSTITUTION_PROFILE, supplies the institution's name, web domains,
people directory, contact address and planning guidance. Without a
profile the institution mode is not offered and no institution-specific
behaviour is active.

See `examples/institution.example.toml` for the format.
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlparse

from src.about import TOOL_NAME, VERSION

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FooterLink:
    label: str
    url: str


@dataclass(frozen=True)
class InstitutionProfile:
    name: str = ""
    short_name: str = ""
    # First entry is the primary domain (used for site: filters).
    domains: tuple[str, ...] = ()
    # Phrases that indicate the institution in free text (case-insensitive).
    keywords: tuple[str, ...] = ()
    openalex_id: str = ""
    contact_email: str = ""
    tool_url: str = ""
    assistant_role: str = ""
    planner_guidelines: str = ""
    directory_name: str = ""
    directory_profile_url: str = ""
    directory_planner_hint: str = ""
    footer_links: tuple[FooterLink, ...] = field(default_factory=tuple)
    # Extra domains never used as sources, e.g. sites that share the
    # institution's abbreviation and pollute its searches.
    blocked_domains: tuple[str, ...] = ()

    # ── derived helpers ─────────────────────────────────────────────

    @property
    def configured(self) -> bool:
        return bool(self.name and self.domains)

    @property
    def label(self) -> str:
        return self.short_name or self.name

    @property
    def primary_domain(self) -> str:
        return self.domains[0] if self.domains else ""

    def site_filter(self) -> str:
        return f"site:{self.primary_domain}" if self.primary_domain else ""

    def strip_site_filter(self, term: str) -> str:
        sf = self.site_filter()
        return term.replace(sf, "").strip() if sf else term.strip()

    def matches_host(self, host: str) -> bool:
        host = (host or "").lower().removeprefix("www.")
        return any(host == d or host.endswith("." + d) for d in self.domains)

    def matches_url(self, url: str) -> bool:
        return self.matches_host(urlparse(url or "").hostname or "")

    def mentions(self, text: str) -> bool:
        rx = self.keyword_regex
        return bool(rx and rx.search(text or ""))

    @property
    def keyword_regex(self) -> re.Pattern | None:
        return _keyword_regex(self.keywords + self.domains)

    def directory_url(self, pid) -> str:
        if not self.directory_profile_url:
            return ""
        return self.directory_profile_url.replace("{pid}", str(pid))


@lru_cache(maxsize=16)
def _keyword_regex(words: tuple[str, ...]) -> re.Pattern | None:
    parts = [re.escape(w).replace(r"\ ", r"\s*") for w in words if w]
    return re.compile("(?:" + "|".join(parts) + ")", re.IGNORECASE) if parts else None


def _parse(data: dict) -> InstitutionProfile:
    inst = data.get("institution", {}) or {}
    directory = data.get("person_directory", {}) or {}
    links = tuple(
        FooterLink(str(l.get("label", "")), str(l.get("url", "")))
        for l in data.get("footer_links", []) or []
        if l.get("label") and l.get("url")
    )
    return InstitutionProfile(
        name=str(inst.get("name", "")).strip(),
        short_name=str(inst.get("short_name", "")).strip(),
        domains=tuple(str(d).strip().lower().removeprefix("www.")
                      for d in inst.get("domains", []) if str(d).strip()),
        keywords=tuple(str(k).strip() for k in inst.get("keywords", []) if str(k).strip()),
        openalex_id=str(inst.get("openalex_id", "")).strip(),
        contact_email=str(inst.get("contact_email", "")).strip(),
        tool_url=str(inst.get("tool_url", "")).strip(),
        assistant_role=str(inst.get("assistant_role", "")).strip(),
        planner_guidelines=str(inst.get("planner_guidelines", "")).strip(),
        directory_name=str(directory.get("name", "")).strip(),
        directory_profile_url=str(directory.get("profile_url", "")).strip(),
        directory_planner_hint=str(directory.get("planner_hint", "")).strip(),
        footer_links=links,
        blocked_domains=tuple(str(d).strip().lower().removeprefix("www.")
                              for d in inst.get("blocked_domains", []) if str(d).strip()),
    )


def load_profile(path: str | None) -> InstitutionProfile:
    """Read a profile file; an empty profile if no path is given.

    A path that is set but unreadable or invalid is a configuration
    error and raises, so a broken profile never silently disables the
    institution features.
    """
    if not path:
        return InstitutionProfile()
    with open(path, "rb") as fh:
        prof = _parse(tomllib.load(fh))
    if not prof.configured:
        raise ValueError(
            f"Institution profile {path}: [institution] needs 'name' and "
            f"at least one entry in 'domains'"
        )
    logger.info("Institution profile loaded: %s (%s)", prof.name, ", ".join(prof.domains))
    return prof


_current: InstitutionProfile | None = None


def get_profile() -> InstitutionProfile:
    """The active profile (loaded once from INSTITUTION_PROFILE)."""
    global _current
    if _current is None:
        _current = load_profile(os.environ.get("INSTITUTION_PROFILE", "").strip() or None)
    return _current


def set_profile(profile: InstitutionProfile | None) -> None:
    """Override the active profile (tests, embedding)."""
    global _current
    _current = profile


def polite_user_agent() -> str:
    """User-Agent for scholarly APIs (Crossref/OpenAlex 'polite pool')."""
    contact = get_profile().contact_email or os.environ.get("CONTACT_EMAIL", "").strip()
    base = f"{TOOL_NAME}/{VERSION}"
    return f"{base} (mailto:{contact})" if contact else base


def tool_url() -> str:
    """Public URL of this installation for report footers ('' = none).

    TOOL_PUBLIC_URL wins over the profile's `tool_url`.
    """
    return os.environ.get("TOOL_PUBLIC_URL", "").strip() or get_profile().tool_url


def operator_line() -> str:
    """'Operated by …' text for exported documents ('' = none)."""
    return get_profile().name
