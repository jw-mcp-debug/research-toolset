"""Institution profile loading and helpers."""

from pathlib import Path

import pytest

from src.institution import InstitutionProfile, load_profile, polite_user_agent, set_profile

EXAMPLE = Path(__file__).resolve().parent.parent / "examples/institution.example.toml"


def test_example_profile_loads():
    p = load_profile(str(EXAMPLE))
    assert p.configured
    assert p.primary_domain == "example.edu"
    assert p.site_filter() == "site:example.edu"
    assert [l.label for l in p.footer_links] == ["Contact", "Imprint", "Privacy"]
    assert p.directory_url(42) == "https://www.example.edu/people?pid=42"


def test_no_profile_is_empty_and_inactive():
    p = load_profile(None)
    assert not p.configured
    assert not p.matches_url("https://www.example.edu/")
    assert p.site_filter() == ""
    assert p.directory_url(1) == ""


def test_domain_matching_includes_subdomains_only():
    p = InstitutionProfile(name="X", domains=("example.edu",))
    assert p.matches_url("https://library.example.edu/a")
    assert p.matches_url("https://www.example.edu/")
    assert not p.matches_url("https://example.edu.evil.org/")
    assert not p.matches_url("https://notexample.edu/")


def test_keywords_and_domains_are_recognised():
    p = InstitutionProfile(name="X", domains=("example.edu",), keywords=("Example University",))
    assert p.mentions("courses at example university")
    assert p.mentions("see example.edu")
    assert not p.mentions("another place")


def test_incomplete_profile_is_rejected(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text('[institution]\nname = "Only a name"\n')
    with pytest.raises(ValueError):
        load_profile(str(f))


def test_polite_user_agent_uses_profile_contact(monkeypatch):
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    set_profile(InstitutionProfile(name="X", domains=("x.org",), contact_email="a@x.org"))
    try:
        assert polite_user_agent().endswith("(mailto:a@x.org)")
    finally:
        set_profile(None)
    set_profile(InstitutionProfile())
    try:
        assert "mailto" not in polite_user_agent()
    finally:
        set_profile(None)


def test_profile_blocked_domains_extend_block_list():
    from src.connectors.base import is_url_blocked
    set_profile(None)
    try:
        set_profile(InstitutionProfile(name="X", domains=("x.org",), blocked_domains=("confusable.example",)))
        assert is_url_blocked("https://www.confusable.example/page")
        assert is_url_blocked("https://sub.confusable.example/")
        set_profile(InstitutionProfile())
        assert not is_url_blocked("https://www.confusable.example/page")
    finally:
        set_profile(None)
