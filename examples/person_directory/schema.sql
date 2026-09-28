-- Schema of the person directory database read by
-- src/connectors/person_directory.py. The tool only reads it (and deletes
-- content of persons whose consent is 0); filling it is the job of an
-- institution-specific sync script that is not part of this repository.

CREATE TABLE persons (
    pid            INTEGER PRIMARY KEY,
    given_name     TEXT NOT NULL DEFAULT '',
    family_name    TEXT NOT NULL,
    academic_title TEXT DEFAULT '',
    status         TEXT DEFAULT '',
    email          TEXT DEFAULT '',
    homepage_url   TEXT DEFAULT '',
    orcid          TEXT DEFAULT '',
    memberships    TEXT DEFAULT '',          -- JSON list of committee names
    consent        INTEGER NOT NULL DEFAULT 0, -- 1 = may be shown
    updated_at     TEXT DEFAULT ''
);

CREATE TABLE orgs (
    org_id    TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    full_path TEXT DEFAULT ''                -- "Faculty → Institute → Unit"
);

CREATE TABLE person_orgs (
    pid             INTEGER NOT NULL REFERENCES persons(pid),
    org_id          TEXT NOT NULL REFERENCES orgs(org_id),
    role            TEXT DEFAULT '',
    subject_area    TEXT DEFAULT '',
    subject_area_en TEXT DEFAULT '',
    phone           TEXT DEFAULT '',
    room            TEXT DEFAULT '',
    building        TEXT DEFAULT '',
    email           TEXT DEFAULT ''
);

CREATE TABLE pages (
    page_id   INTEGER PRIMARY KEY,
    pid       INTEGER REFERENCES persons(pid),  -- NULL = organisation page
    url       TEXT DEFAULT '',
    title     TEXT DEFAULT '',
    content   TEXT DEFAULT '',
    page_type TEXT DEFAULT ''
);

-- float32 vectors (numpy .tobytes()), produced with EMBEDDER_MODEL
CREATE TABLE embeddings (
    page_id   INTEGER PRIMARY KEY REFERENCES pages(page_id),
    embedding BLOB NOT NULL
);

CREATE VIRTUAL TABLE search_index USING fts5(
    pid, page_id, person_name, org_path, subject_area, title, content
);

CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT DEFAULT ''
);
