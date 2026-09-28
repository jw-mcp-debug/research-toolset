# Output languages

The user interface is English. The *output language* — the language of
reports and exports — is chosen per run in the interface.

## What follows the output language

- Everything the tool itself writes into a report or export: headings, the
  report header and footer, notes and warning banners, the bibliography check
  report, Word export appendices, the search protocol of the analysis modes,
  date formats. These texts come from catalogs.
- Everything the model writes for the reader: the report, summaries,
  analysis sections, diagnosis messages. Prompts are English; each prompt whose
  output reaches the report ends with an instruction naming the output
  language (the catalog's `llm_name`).

Search queries are not affected: the planner chooses search languages by
topic, and literature queries stay English because the scholarly APIs work
best with English terms.

## Enabling languages

```bash
OUTPUT_LANGUAGES=en,de          # languages offered in the interface
DEFAULT_OUTPUT_LANGUAGE=de      # preselected
```

English is always available. With only one language enabled, the selector is
hidden.

## Catalogs

A catalog is a TOML file named after the language code:

```toml
[meta]
name = "Deutsch"          # shown in the interface
llm_name = "German"       # used in the instruction to the model

[strings]
"footer.title" = "Hinweise zur Erstellung"
"search.n_hits" = "**{n} Treffer**"
# ...
```

- `src/locales/en.toml` is the reference: every key the code uses exists there.
- `src/locales/de.toml` ships with the tool.
- Keys missing from a catalog fall back to English, so a partial catalog is
  usable.
- Placeholders such as `{n}` must match the English entry exactly; a date
  format key (`footer.datetime_format`, `word.date_format`) takes a
  `strftime` pattern.

At start-up every enabled catalog is validated: an unknown language, a
missing catalog or a placeholder mismatch stops the start with a message that
names the key.

## Adding a language

1. Copy `src/locales/en.toml` to `<code>.toml` (e.g. `nl.toml`) in a directory
   of your choice.
2. Set `name` and `llm_name` and translate the strings you need.
3. Point `OUTPUT_CATALOG_DIR` at the directory and add the code to
   `OUTPUT_LANGUAGES`.

No code change is needed. A catalog in `OUTPUT_CATALOG_DIR` takes precedence
over a built-in one with the same code, so you can also adapt the German
wording this way.

## What stays German on purpose

Some German text in the code is data, not interface: German stop words and
titles used to detect languages and names, German headings recognised in
bibliographies, and German spellings the parsers still accept when a model
answers in German.
