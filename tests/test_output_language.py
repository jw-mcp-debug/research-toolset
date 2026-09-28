"""Output languages: settings, catalogs, fallback and validation."""

import pytest

from src import output_language as ol


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for k in ("OUTPUT_LANGUAGES", "DEFAULT_OUTPUT_LANGUAGE", "OUTPUT_CATALOG_DIR"):
        monkeypatch.delenv(k, raising=False)
    ol.reload()
    yield
    ol.reload()


def test_default_is_english_only():
    assert ol.enabled_languages() == ("en",)
    assert ol.default_language() == "en"
    assert ol.language_choices() == [("English", "en")]


def test_german_catalog_is_built_in(monkeypatch):
    monkeypatch.setenv("OUTPUT_LANGUAGES", "de")
    ol.reload()
    assert ol.enabled_languages() == ("en", "de")        # English always offered
    assert ol.t("search.n_hits", "de", n=3) == "**3 Treffer**"
    assert ol.llm_language_name("de") == "German"


def test_default_language_setting(monkeypatch):
    monkeypatch.setenv("OUTPUT_LANGUAGES", "en,de")
    monkeypatch.setenv("DEFAULT_OUTPUT_LANGUAGE", "de")
    ol.reload()
    assert ol.default_language() == "de"
    assert ol.normalize("fr") == "de"                    # not enabled → default


def test_extra_catalog_from_directory_and_fallback(monkeypatch, tmp_path):
    (tmp_path / "nl.toml").write_text(
        '[meta]\nname = "Nederlands"\nllm_name = "Dutch"\n[strings]\n', encoding="utf-8")
    monkeypatch.setenv("OUTPUT_LANGUAGES", "en,nl")
    monkeypatch.setenv("OUTPUT_CATALOG_DIR", str(tmp_path))
    ol.reload()
    assert ("Nederlands", "nl") in ol.language_choices()
    assert ol.llm_language_name("nl") == "Dutch"
    # key missing in nl → English
    assert ol.t("search.n_hits", "nl", n=2) == "**2 hits**"


def test_unknown_language_fails_at_start(monkeypatch):
    monkeypatch.setenv("OUTPUT_LANGUAGES", "en,xx")
    ol.reload()
    with pytest.raises(ValueError, match="xx"):
        ol.enabled_languages()


def test_catalog_with_wrong_placeholder_is_rejected(monkeypatch, tmp_path):
    (tmp_path / "nl.toml").write_text(
        '[meta]\nname = "Nederlands"\n[strings]\n"search.n_hits" = "**{aantal} treffers**"\n',
        encoding="utf-8")
    monkeypatch.setenv("OUTPUT_LANGUAGES", "nl")
    monkeypatch.setenv("OUTPUT_CATALOG_DIR", str(tmp_path))
    ol.reload()
    with pytest.raises(ValueError, match="placeholders"):
        ol.enabled_languages()


def test_builtin_catalogs_are_consistent():
    """Every shipped catalog passes the same validation as a custom one."""
    core = ol._read(ol._BUILTIN_DIR / "en.toml")
    for path in ol._BUILTIN_DIR.glob("*.toml"):
        assert ol.validate_catalog(ol._read(path), core) == [], path.name


def test_orchestrator_passes_language_into_context():
    from src.pipeline.orchestrator import ResearchOrchestrator
    orch = ResearchOrchestrator(None, None, None, output_language="xx")
    assert orch.output_language == "en"                  # not enabled → default
