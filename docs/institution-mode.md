# Institution mode

The institution mode focuses a research run on one organisation — a
university, a company, an agency: its own web pages, its search index and its
person directory. Everything that ties the tool to one organisation lives in
an *institution profile*; without a profile the mode is not offered and the
tool behaves the same for everyone.

## Institution profile

Copy [`examples/institution.example.toml`](../examples/institution.example.toml)
and point `INSTITUTION_PROFILE` at the copy.

| Field | Used for |
|---|---|
| `name`, `short_name` | prompts and reports use the name; the mode label ("… research") uses the short name if set |
| `domains` | the first domain is used for `site:` filters; subdomains count as the institution's own |
| `keywords` | recognising the institution in search terms |
| `openalex_id` | OpenAlex look-ups restricted to the institution (authors, topics); empty = off |
| `contact_email` | User-Agent for Crossref and OpenAlex |
| `blocked_domains` | sites never used as sources, e.g. sites that share the institution's abbreviation |
| `tool_url` | printed in report footers (overridden by `TOOL_PUBLIC_URL`) |
| `assistant_role` | first line of the chat assistant's system prompt |
| `planner_guidelines` | extra planning rules for the institution mode (write them in English, like the prompts) |
| `[person_directory]` | `name`, `profile_url` (with `{pid}`), optional `planner_hint` |
| `[[footer_links]]` | links shown in the interface footer (imprint, privacy, contact) |

In the interface the mode offers an *"… only"* option that restricts every
search to the institution's domains and disables external connectors.

## Person directory

With `PERSON_DIRECTORY_DB` pointing at an SQLite database, the institution
mode can search the institution's people and units. The schema is in
[`examples/person_directory/schema.sql`](../examples/person_directory/schema.sql);
[`build_example_db.py`](../examples/person_directory/build_example_db.py)
builds a small fictional database to try it:

```bash
python examples/person_directory/build_example_db.py data/example_directory.db
PERSON_DIRECTORY_DB=data/example_directory.db
```

The search combines full-text search (FTS5), vector similarity (with an
embedder, see `EMBEDDER_*`), a reranker (`RERANKER_*`) and a hard name filter:
if a query looks like a person's name, every name part must occur in the
result, so a search for one person does not return others who share a given
name.

**Consent.** Only persons with `consent = 1` are ever returned. When the
connector is first used, the content of persons with `consent = 0` is deleted
from the searchable tables (pages, embeddings, full-text index). Maintaining
the `consent` column — and filling the database at all — is the job of an
institution-specific synchronisation job, which is not part of this
repository. Apart from that clean-up (and adding a missing `consent`
column, defaulting to 0) the tool only reads the database.

## Website search index

Two optional connectors search an index of the institution's own website:

- **Solr** (`SOLR_SCHEME`, `SOLR_HOST`, `SOLR_PORT`, `SOLR_CORE_DE`,
  `SOLR_CORE_EN`, credentials) — two cores, one per language; in the
  institution mode every search term is also sent to Solr. On the first
  research run the configured field names are checked against the index.
- **Elasticsearch** (`ELASTIC_*`) — one index with a configurable field
  mapping; used as the `elastic` source scope.

Both are optional; the institution mode works with web search alone.
