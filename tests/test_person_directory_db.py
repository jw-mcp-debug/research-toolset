"""Person directory connector against the fictional example database."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

from src.connectors.person_directory import DirectorySearchConfig, PersonDirectoryConnector
from src.institution import InstitutionProfile, set_profile

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_example_db", ROOT / "examples/person_directory/build_example_db.py")
build_example_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_example_db)


@pytest.fixture
def directory(tmp_path):
    db = tmp_path / "dir.db"
    build_example_db.build(str(db))
    set_profile(InstitutionProfile(
        name="Example University", domains=("example.edu",),
        directory_profile_url="https://www.example.edu/people?pid={pid}"))
    conn = PersonDirectoryConnector(DirectorySearchConfig(db_path=str(db)))
    yield conn, db
    conn.close()
    set_profile(None)


async def test_person_query_filters_same_first_name(directory):
    conn, _ = directory
    hits = await conn.search("Mira Okonkwo")
    assert hits, "expected a hit"
    assert {h.person_name for h in hits} == {"Mira Okonkwo"}
    top = hits[0]
    assert top.academic_title == "Prof. Dr."
    assert top.role == "Head of institute"
    assert top.building == "Main building"


async def test_person_without_consent_is_never_returned(directory):
    conn, _ = directory
    assert await conn.search("Jonas Weller") == []
    assert conn.verify_person("Jonas Weller") is None
    assert conn.get_person_details(4) is None


async def test_revoked_consent_purges_searchable_content(directory):
    conn, db = directory
    await conn.ensure_consent_fresh()
    raw = sqlite3.connect(db)
    assert raw.execute("SELECT COUNT(*) FROM pages WHERE pid=4").fetchone()[0] == 0
    assert raw.execute("SELECT COUNT(*) FROM search_index WHERE pid=4").fetchone()[0] == 0
    # persons with consent keep their content
    assert raw.execute("SELECT COUNT(*) FROM pages WHERE pid=1").fetchone()[0] == 1


def test_details_and_profile_url(directory):
    conn, _ = directory
    d = conn.get_person_details(1)
    assert d["person"]["family_name"] == "Okonkwo"
    assert d["orgs"][0]["role"] == "Head of institute"


async def test_markdown_uses_profile_url(directory):
    conn, _ = directory
    hit = (await conn.search("Tomas Brandt"))[0]
    md = hit.to_markdown()
    assert "Tomas Brandt" in md
    assert "https://www.example.edu/people/brandt" in md  # homepage wins


def test_missing_consent_column_is_added_as_not_consented(tmp_path):
    db = tmp_path / "old.db"
    raw = sqlite3.connect(db)
    raw.execute("CREATE TABLE persons (pid INTEGER PRIMARY KEY, family_name TEXT)")
    raw.commit(); raw.close()
    conn = PersonDirectoryConnector(DirectorySearchConfig(db_path=str(db)))
    c = conn._get_conn()
    c.execute("INSERT INTO persons (pid, family_name) VALUES (1, 'X')")
    assert c.execute("SELECT consent FROM persons WHERE pid=1").fetchone()[0] == 0
    conn.close()
