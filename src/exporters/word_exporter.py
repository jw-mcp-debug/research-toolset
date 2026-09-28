"""
Word export of research reports.

Converts the Markdown report into a structured Word document with real
hyperlinks, inline formatting and appendices (sources, extracts,
progress, metadata).
"""

from src.about import TOOL_NAME, VERSION
import logging
import os
import re
import tempfile
from datetime import datetime

logger = logging.getLogger(__name__)



import contextvars

from src.output_language import t as _catalog_t

# Output language of the export in progress (set by export_research_to_word).
_EXPORT_LANG: contextvars.ContextVar[str] = contextvars.ContextVar("export_lang", default="en")


def _t(key: str, **values) -> str:
    """Catalog lookup in the output language of the current export."""
    return _catalog_t(key, _EXPORT_LANG.get(), **values)

# ─── Hyperlink helper (python-docx has no native API for this) ─────

def _add_hyperlink(paragraph, url: str, text: str, font_size=None, color="2563EB"):
    """Insert a clickable hyperlink into a Word paragraph."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    # Create the relationship
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )

    # XML elements
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    run_el = OxmlElement("w:r")

    # Formatting
    rPr = OxmlElement("w:rPr")

    c_el = OxmlElement("w:color")
    c_el.set(qn("w:val"), color)
    rPr.append(c_el)

    u_el = OxmlElement("w:u")
    u_el.set(qn("w:val"), "single")
    rPr.append(u_el)

    if font_size:
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(font_size * 2))  # half points
        rPr.append(sz)

    style_el = OxmlElement("w:rStyle")
    style_el.set(qn("w:val"), "Hyperlink")
    rPr.append(style_el)

    run_el.append(rPr)

    text_el = OxmlElement("w:t")
    text_el.set(qn("xml:space"), "preserve")
    text_el.text = text
    run_el.append(text_el)

    hyperlink.append(run_el)
    paragraph._p.append(hyperlink)

    return hyperlink


# ─── Inline-Markdown-Parser ───────────────────────────────────────

_INLINE_RE = re.compile(
    r"(?P<bold_italic>\*\*\*(.+?)\*\*\*)"
    r"|(?P<bold>\*\*(.+?)\*\*)"
    r"|(?P<italic>\*(.+?)\*)"
    r"|(?P<code>`(.+?)`)"
    r"|(?P<link>\[([^\]]+)\]\(([^\)]+)\))"
    r"|(?P<text>[^*`\[]+)"
    r"|(?P<other>.)",
    re.DOTALL,
)


def _add_inline_markdown(paragraph, text: str, base_size=None, base_bold=False):
    """Parse inline Markdown and create formatted runs in the paragraph."""
    from docx.shared import Pt, RGBColor

    # group indices: bold_italic=2, bold=4, italic=6, code=8, link_text=10, link_url=11
    for m in _INLINE_RE.finditer(text):
        if m.group("bold_italic"):
            content = m.group(2)
            run = paragraph.add_run(content)
            run.bold = True
            run.italic = True
            if base_size:
                run.font.size = Pt(base_size)

        elif m.group("bold"):
            content = m.group(4)
            run = paragraph.add_run(content)
            run.bold = True
            if base_size:
                run.font.size = Pt(base_size)

        elif m.group("italic"):
            content = m.group(6)
            run = paragraph.add_run(content)
            run.italic = True
            if base_size:
                run.font.size = Pt(base_size)

        elif m.group("code"):
            content = m.group(8)
            run = paragraph.add_run(content)
            run.font.name = "Consolas"
            run.font.size = Pt((base_size or 10) - 1)
            run.font.color.rgb = RGBColor(180, 60, 30)

        elif m.group("link"):
            link_text = m.group(10)
            link_url = m.group(11).strip().rstrip("]}).,:;'\"")
            _add_hyperlink(paragraph, link_url, link_text,
                           font_size=base_size)

        elif m.group("text") or m.group("other"):
            content = m.group("text") or m.group("other")
            run = paragraph.add_run(content)
            if base_bold:
                run.bold = True
            if base_size:
                run.font.size = Pt(base_size)


# ─── Markdown table → Word ──────────────────────────────────────

def _parse_table_row(line: str) -> list[str]:
    """Parse a Markdown table row into cells."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _is_separator_row(line: str) -> bool:
    """Check whether a line is the table separator row (|---|---| or ---|---)."""
    cells = _parse_table_row(line)
    if not cells or not any(c.strip() for c in cells):
        return False
    return all(re.match(r"^:?-{2,}:?$", c.strip()) for c in cells if c.strip())


def _looks_like_table_start(line: str, next_line: str | None) -> bool:
    """Check whether a line starts a Markdown table.

    Recognises both formats:
      | Col1 | Col2 |      (with a leading pipe)
      Col1 | Col2           (without a leading pipe)

    Condition: the next line must be a separator row (---|---).
    """
    stripped = line.strip()
    if "|" not in stripped:
        return False
    # Must have at least 2 cells
    cells = _parse_table_row(stripped)
    if len(cells) < 2:
        return False
    # The next line must be a separator row
    if next_line and _is_separator_row(next_line):
        return True
    return False


def _collect_table_lines(lines: list[str], start: int) -> list[str]:
    """Collect all contiguous table rows from position start."""
    table_lines = []
    i = start
    # First line (header)
    table_lines.append(lines[i].strip())
    i += 1

    while i < len(lines):
        stripped = lines[i].strip()
        # Empty line → the table ends
        if not stripped:
            break
        # separator or data row with | → belongs to the table
        if "|" in stripped:
            table_lines.append(stripped)
            i += 1
        else:
            break

    return table_lines


def _add_markdown_table(doc, table_lines: list[str]):
    """Convert Markdown table rows into a Word table."""
    from docx.shared import Pt
    from docx.enum.table import WD_TABLE_ALIGNMENT

    # Parse the header
    headers = _parse_table_row(table_lines[0])
    num_cols = len(headers)

    # Skip the separator row, collect the data rows
    data_rows = []
    for line in table_lines[1:]:
        if _is_separator_row(line):
            continue
        cells = _parse_table_row(line)
        # Pad/truncate to the same number of columns
        while len(cells) < num_cols:
            cells.append("")
        cells = cells[:num_cols]
        data_rows.append(cells)

    if not num_cols:
        return

    # Create the Word table
    table = doc.add_table(rows=1 + len(data_rows), cols=num_cols)
    table.style = "Light Shading"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Header row
    hdr_cells = table.rows[0].cells
    for j, header_text in enumerate(headers):
        hdr_cells[j].text = ""  # clear the default text
        p = hdr_cells[j].paragraphs[0]
        _add_inline_markdown(p, header_text, base_size=9, base_bold=True)

    # Data rows
    for row_idx, row_data in enumerate(data_rows):
        row_cells = table.rows[row_idx + 1].cells
        for j, cell_text in enumerate(row_data):
            row_cells[j].text = ""
            p = row_cells[j].paragraphs[0]
            _add_inline_markdown(p, cell_text, base_size=9)

    # Set cell spacing
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                pf = paragraph.paragraph_format
                pf.space_before = Pt(2)
                pf.space_after = Pt(2)

    # Some space after the table
    doc.add_paragraph()


# ─── Markdown → Word converter ────────────────────────────────────

def _add_markdown_to_doc(doc, text: str):
    """Convert Markdown text into formatted Word paragraphs."""
    from docx.shared import Pt, RGBColor
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    lines = text.split("\n")
    i = 0
    in_code_block = False
    code_buffer = []

    while i < len(lines):
        line = lines[i]

        # Start/end of a code block
        if line.strip().startswith("```"):
            if in_code_block:
                code_text = "\n".join(code_buffer)
                if code_text.strip():
                    p = doc.add_paragraph()
                    run = p.add_run(code_text)
                    run.font.name = "Consolas"
                    run.font.size = Pt(8)
                    pf = p.paragraph_format
                    pf.space_before = Pt(4)
                    pf.space_after = Pt(4)
                    shading = OxmlElement("w:shd")
                    shading.set(qn("w:fill"), "F3F4F6")
                    shading.set(qn("w:val"), "clear")
                    p._p.get_or_add_pPr().append(shading)
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
                code_buffer = []
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue

        stripped = line.strip()

        # Empty line
        if not stripped:
            i += 1
            continue

        # Headings
        heading_match = re.match(r"^(#{1,4})\s+(.+)", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            doc.add_heading(heading_match.group(2), level=level)
            i += 1
            continue

        # Detect Markdown tables (BEFORE detecting ---, otherwise ---|---
        # would be taken as a horizontal rule)
        # Format 1: | Col1 | Col2 |  (with a leading pipe)
        # Format 2: Col1 | Col2      (without a leading pipe)
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else None
        if _looks_like_table_start(stripped, next_line):
            table_lines = _collect_table_lines(lines, i)
            i += len(table_lines)
            if len(table_lines) >= 2:
                _add_markdown_table(doc, table_lines)
            else:
                p = doc.add_paragraph()
                _add_inline_markdown(p, stripped)
            continue

        # Horizontal rule (only a real --- without a pipe)
        if (stripped.startswith("---") or stripped.startswith("***")) and "|" not in stripped:
            p = doc.add_paragraph()
            run = p.add_run("─" * 60)
            run.font.size = Pt(6)
            run.font.color.rgb = RGBColor(180, 180, 180)
            i += 1
            continue

        # Bullet list (- or *)
        if re.match(r"^[-*]\s+", stripped):
            p = doc.add_paragraph(style="List Bullet")
            _add_inline_markdown(p, re.sub(r"^[-*]\s+", "", stripped))
            i += 1
            continue

        # Numbered list
        num_match = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if num_match:
            p = doc.add_paragraph(style="List Number")
            _add_inline_markdown(p, num_match.group(2))
            i += 1
            continue

        # Blockquote
        if stripped.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(36)
            _add_inline_markdown(p, stripped[2:], base_size=10)
            if p.runs:
                p.runs[0].italic = True
            i += 1
            continue

        # Normal paragraph — collect consecutive lines
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            next_line = lines[i].strip()
            if not next_line:
                break
            if re.match(r"^(#{1,4}\s|[-*]\s|>\s|```|---|\*\*\*|\d+\.\s)", next_line):
                break
            para_lines.append(next_line)
            i += 1

        full_para = " ".join(para_lines)
        p = doc.add_paragraph()
        _add_inline_markdown(p, full_para)


# ─── Appendix sections ────────────────────────────────────────────

def _add_sources_appendix(doc, sources: list):
    """Appendix A: list of sources."""
    from docx.shared import Pt, RGBColor
    from urllib.parse import urlparse
    from src.connectors.base import clean_url

    doc.add_heading(_t("word.appendix_sources"), level=1)

    if not sources:
        doc.add_paragraph(_t("word.no_sources"))
        return

    type_labels = {
        "web_search": _t("word.group.web_search"),
        "web_page": _t("word.group.web_pages"),
        "git_repo": "Git-Repositories",
        "git_file": _t("word.group.git_files"),
        "git_issue": "Git-Issues",
        "local_file": _t("word.group.local_files"),
        "elastic": "Website-Index",
        "directory_person": _t("word.group.directory_persons"),
        "directory_org": _t("word.group.directory_units"),
    }
    groups = {}
    for s in sources:
        type_key = s.source_type.value if hasattr(s.source_type, "value") else str(s.source_type)
        label = type_labels.get(type_key, _t("word.group.other"))
        if label not in groups:
            groups[label] = []
        groups[label].append(s)

    total_idx = 1
    for label, items in groups.items():
        doc.add_heading(f"{label} ({len(items)})", level=2)

        for s in items:
            url = clean_url(s.url) if s.url else ""
            try:
                domain = urlparse(url).hostname or ""
                domain = domain.removeprefix("www.")
            except Exception:
                domain = ""

            p = doc.add_paragraph()
            run_num = p.add_run(f"[{total_idx}] ")
            run_num.bold = True
            run_num.font.size = Pt(9)

            run_title = p.add_run(f"{s.title}\n")
            run_title.font.size = Pt(9)

            if url:
                _add_hyperlink(p, url, url, font_size=8)

            if domain:
                run_domain = p.add_run(f"  ({domain})")
                run_domain.font.size = Pt(8)
                run_domain.font.color.rgb = RGBColor(128, 128, 128)

            if hasattr(s, "content_length") and s.content_length:
                meta_parts = [_t("word.characters", n=f"{s.content_length:,}")]
                if hasattr(s, "fetch_time_seconds") and s.fetch_time_seconds:
                    meta_parts.append(f"{s.fetch_time_seconds:.1f}s")
                run_meta = p.add_run(f"\n{' · '.join(meta_parts)}")
                run_meta.font.size = Pt(7)
                run_meta.font.color.rgb = RGBColor(160, 160, 160)

            total_idx += 1


def _add_extracts_appendix(doc, harvest_results: list, extracts: list):
    """Appendix B: extracts."""
    from docx.shared import Pt, RGBColor

    doc.add_heading(_t("word.appendix_extracts"), level=1)

    if not extracts:
        doc.add_paragraph(_t("word.no_extracts"))
        return

    by_question = {}
    for e in extracts:
        qid = e.question_id
        if qid not in by_question:
            by_question[qid] = []
        by_question[qid].append(e)

    for qid, items in sorted(by_question.items()):
        doc.add_heading(_t("word.question", qid=qid), level=2)

        for e in items:
            p = doc.add_paragraph(style="List Bullet")
            run_fact = p.add_run(e.fact)
            run_fact.bold = True
            run_fact.font.size = Pt(9)

            src_title = (e.source_title or e.source_url or "")[:60]
            src_url = e.source_url.strip().rstrip("]}).,:;'\"") if e.source_url else ""
            p.add_run("\n").font.size = Pt(9)
            run_src_label = p.add_run(_t("word.source_prefix"))
            run_src_label.font.size = Pt(8)
            run_src_label.font.color.rgb = RGBColor(128, 128, 128)
            if src_url:
                _add_hyperlink(p, src_url, src_title, font_size=8)
            else:
                run_src = p.add_run(src_title)
                run_src.font.size = Pt(8)

            rel_colors = {
                "high": RGBColor(22, 163, 74),
                "medium": RGBColor(202, 138, 4),
                "low": RGBColor(220, 38, 38),
            }
            p.add_run(" · ").font.size = Pt(8)
            run_rel = p.add_run(_t("word.reliability", value=e.reliability))
            run_rel.font.size = Pt(8)
            run_rel.font.color.rgb = rel_colors.get(e.reliability,
                                                     RGBColor(128, 128, 128))

            if e.context:
                run_ctx = p.add_run(f"\n{e.context}")
                run_ctx.font.size = Pt(8)
                run_ctx.font.color.rgb = RGBColor(100, 100, 100)
                run_ctx.italic = True


def _add_progress_appendix(doc, progress_log: list, ctx):
    """Appendix C: research log."""
    from docx.shared import Pt

    doc.add_heading(_t("word.appendix_progress"), level=1)

    p = doc.add_paragraph()
    summary_lines = [
        _t("word.request", query=ctx.query),
        _t("word.started", ts=_format_timestamp(ctx.started_at)),
        _t("word.finished", ts=_format_timestamp(ctx.finished_at) if ctx.finished_at else _t("word.not_finished")),
        _t("word.duration_seconds", n=f"{ctx.duration_seconds:.0f}"),
        _t("word.rounds", n=ctx.rounds_completed),
        _t("word.sources_count", n=len(ctx.sources)),
        _t("word.extracts_count", n=len(ctx.extracts)),
        f"Status: {ctx.status}",
    ]
    for line in summary_lines:
        run = p.add_run(f"{line}\n")
        run.font.size = Pt(9)

    # Search strategy
    if ctx.search_stats:
        lang_names = {
            "de": _t("lang.de"), "en": _t("lang.en"), "zh": _t("lang.zh"),
            "pt": _t("lang.pt"), "es": _t("lang.es"), "fr": _t("lang.fr"),
            "ja": _t("lang.ja"), "ko": _t("lang.ko"), "ru": _t("lang.ru"),
            "it": _t("lang.it"), "nl": _t("lang.nl"), "ar": _t("lang.ar"),
            "pl": _t("lang.pl"), "tr": _t("lang.tr"),
        }
        doc.add_heading(_t("word.search_strategy"), level=2)

        langs = ctx.search_stats.get("languages", [])
        p_langs = doc.add_paragraph()
        lang_display = ", ".join(lang_names.get(l, l) for l in langs)
        run = p_langs.add_run(_t("word.search_languages", langs=lang_display))
        run.font.size = Pt(9)
        run.bold = True

        terms_by_lang = ctx.search_stats.get("terms_by_lang", {})
        for lang, terms in terms_by_lang.items():
            name = lang_names.get(lang, lang)
            p_lang = doc.add_paragraph()
            run_header = p_lang.add_run(f"{name}:")
            run_header.font.size = Pt(9)
            run_header.bold = True
            for term in terms:
                run_term = p_lang.add_run(f"\n  • {term}")
                run_term.font.size = Pt(8)

    if not progress_log:
        return

    doc.add_heading(_t("word.progress_log"), level=2)

    table = doc.add_table(rows=1, cols=3)
    table.style = "Light Shading"

    hdr = table.rows[0].cells
    for i, text in enumerate([_t("word.time"), "Phase", _t("word.message")]):
        hdr[i].text = text
        for run in hdr[i].paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(8)

    for entry in progress_log:
        row = table.add_row().cells
        ts = _format_timestamp(entry.timestamp) if hasattr(entry, "timestamp") else ""
        phase = entry.phase.value if hasattr(entry.phase, "value") else str(entry.phase)
        msg = entry.message[:100] if hasattr(entry, "message") else ""

        for i, text in enumerate([ts, phase, msg]):
            row[i].text = text
            for run in row[i].paragraphs[0].runs:
                run.font.size = Pt(7)


def _add_pipeline_run_appendix(doc, ctx):
    """Appendix D: pipeline run (full process transparency).

    Reuses the UI renderer `render_pipeline_run` (returns clean
    Markdown) and the existing Markdown→Word converter, so the Word
    appendix automatically stays consistent with the tab — a single
    source of truth.

    FAIL-OPEN: an error while rendering must NEVER break the whole Word
    export (the report matters more than the appendix).
    """
    from docx.shared import Pt

    doc.add_heading(_t("word.appendix_pipeline"), level=1)

    p_intro = doc.add_paragraph()
    run = p_intro.add_run(
        _t("word.pipeline_intro")
    )
    run.font.size = Pt(9)
    run.italic = True

    try:
        from src.ui.pipeline_run import render_pipeline_run
        md = render_pipeline_run(ctx)
    except Exception as e:  # pragma: no cover - protective path
        logger.warning("Pipeline-run appendix: rendering failed: %s", e)
        doc.add_paragraph(
            _t("word.pipeline_unavailable")
        )
        return

    if not md or not str(md).strip():
        doc.add_paragraph(_t("word.pipeline_none"))
        return

    # render_pipeline_run often returns its own top heading
    # (e.g. "# 🧠 Pipeline run"). Remove the first H1 so that the Word
    # document has no duplicate heading next to "Appendix D".
    text = str(md).lstrip()
    if text.startswith("# "):
        nl = text.find("\n")
        text = text[nl:].lstrip("\n") if nl > 0 else ""

    try:
        _add_markdown_to_doc(doc, text)
    except Exception as e:  # pragma: no cover - protective path
        logger.warning(
            "Pipeline-run appendix: Markdown→Word failed: %s", e
        )
        # Last resort: raw text, so that the content is not lost
        p = doc.add_paragraph()
        r = p.add_run(text[:20000])
        r.font.size = Pt(8)


def _add_contradictions_appendix(doc, contradictions: list):
    """Contradictions between sources."""
    from docx.shared import Pt

    doc.add_heading(_t("word.contradictions"), level=1)

    p = doc.add_paragraph()
    run = p.add_run(
        _t("word.contradictions_intro", n=len(contradictions))
    )
    run.font.size = Pt(10)

    for i, c in enumerate(contradictions, 1):
        nature = c.get("nature", _t("word.contradiction_unknown"))
        doc.add_heading(_t("word.contradiction_n", i=i, nature=nature), level=2)

        fact_a = c.get("fact_a", "?")
        source_a = c.get("source_a", "?")
        fact_b = c.get("fact_b", "?")
        source_b = c.get("source_b", "?")

        p = doc.add_paragraph()
        run = p.add_run(_t("word.source_a"))
        run.bold = True
        run.font.size = Pt(9)
        run = p.add_run(f"{source_a}")
        run.font.size = Pt(9)

        p = doc.add_paragraph()
        run = p.add_run(f"→ {fact_a}")
        run.font.size = Pt(9)
        run.italic = True

        p = doc.add_paragraph()
        run = p.add_run(_t("word.source_b"))
        run.bold = True
        run.font.size = Pt(9)
        run = p.add_run(f"{source_b}")
        run.font.size = Pt(9)

        p = doc.add_paragraph()
        run = p.add_run(f"→ {fact_b}")
        run.font.size = Pt(9)
        run.italic = True


def _add_metadata_appendix(doc, ctx):
    """Appendix E: metadata (tool URL, LLM, transparency)."""
    from docx.shared import Pt, RGBColor

    doc.add_heading(_t("word.appendix_metadata"), level=1)

    p_intro = doc.add_paragraph()
    run = p_intro.add_run(
        _t("word.metadata_intro")
    )
    run.font.size = Pt(9)
    run.italic = True

    # Tool
    doc.add_heading("Tool", level=2)
    p = doc.add_paragraph()
    run = p.add_run(f"{TOOL_NAME} {VERSION}\n")
    run.bold = True
    run.font.size = Pt(9)

    from src.institution import operator_line, tool_url as _tool_url
    url = _tool_url()
    if url:
        run_label = p.add_run("URL: ")
        run_label.font.size = Pt(9)
        _add_hyperlink(p, url, url, font_size=9)
        p.add_run("\n")
    operator = operator_line()
    if operator:
        run_inst = p.add_run(_t("word.operator", operator=operator))
        run_inst.font.size = Pt(9)

    # LLM
    doc.add_heading(_t("word.llm_used"), level=2)
    p = doc.add_paragraph()

    model_name = os.environ.get("LLM_MODEL_NAME", _t("footer.unknown"))
    api_base = os.environ.get("LLM_API_BASE", "")
    harvest_model = os.environ.get("HARVEST_LLM_MODEL_NAME", model_name)
    harvest_base = os.environ.get("HARVEST_LLM_API_BASE", api_base)

    # Token statistics from ctx.llm_usage (if present)
    llm_usage = getattr(ctx, "llm_usage", {}) or {}
    p_usage = llm_usage.get("primary", {})
    h_usage = llm_usage.get("harvest", {})

    # Prefer model names from the usage data (more current than env vars)
    if p_usage.get("model"):
        model_name = p_usage["model"]
    if h_usage.get("model"):
        harvest_model = h_usage["model"]

    entries = [
        (_t("word.primary_llm"), model_name, api_base, p_usage),
        (_t("word.harvest_llm"), harvest_model, harvest_base, h_usage),
    ]
    for label, model, base, usage_data in entries:
        run_label = p.add_run(f"{label}:\n")
        run_label.font.size = Pt(9)
        run_label.bold = True
        detail = _t("word.model_endpoint", model=model, base=base)
        if usage_data:
            reqs = usage_data.get("requests", 0)
            prompt_t = usage_data.get("prompt_tokens", 0)
            compl_t = usage_data.get("completion_tokens", 0)
            total_t = usage_data.get("total_tokens", 0)
            detail += (
                _t("word.requests_tokens", reqs=f"{reqs:,}", total=f"{total_t:,}", prompt=f"{prompt_t:,}", compl=f"{compl_t:,}")
            )
        run_detail = p.add_run(f"{detail}\n")
        run_detail.font.size = Pt(8)

    # Overall statistics
    if llm_usage.get("total_requests"):
        p_total = doc.add_paragraph()
        run_t = p_total.add_run(
            _t("word.total_requests_tokens", reqs=f"{llm_usage['total_requests']:,}", tokens=f"{llm_usage.get('total_tokens', 0):,}")
        )
        run_t.font.size = Pt(8)
        run_t.bold = True

    # Research parameters
    doc.add_heading(_t("word.research_parameters"), level=2)
    p = doc.add_paragraph()

    is_lit_check = (
        ctx.output_schema
        and ctx.output_schema.format_type == "literature_check"
    )

    if is_lit_check and ctx.search_stats and ctx.search_stats.get("type") == "literature_check":
        # Literature check: API and LLM statistics
        api_stats = ctx.search_stats.get("api_stats", {})
        llm_stats = ctx.search_stats.get("llm_stats", {})
        api_calls = api_stats.get("api_calls", {})
        api_hits = api_stats.get("api_hits", {})

        params = [
            _t("word.research_id", id=ctx.id),
            _t("word.mode_litcheck"),
            _t("word.entries_checked", n=api_stats.get('entries_total', 0)),
            _t("word.duration_s", n=f"{ctx.duration_seconds:.0f}"),
            "",
            _t("word.api_queries_heading"),
            _t("word.api_line", api="arXiv", calls=api_calls.get("arxiv", 0), hits=api_hits.get("arxiv", 0)),
            _t("word.api_line", api="CrossRef", calls=api_calls.get("crossref", 0), hits=api_hits.get("crossref", 0)),
            _t("word.api_line", api="OpenAlex", calls=api_calls.get("openalex", 0), hits=api_hits.get("openalex", 0)),
            _t("word.api_line", api="Semantic Scholar", calls=api_calls.get("semantic_scholar", 0), hits=api_hits.get("semantic_scholar", 0)),
            _t("word.api_line", api="DBLP", calls=api_calls.get("dblp", 0), hits=api_hits.get("dblp", 0)),
            _t("word.api_line", api=_t("word.group.web_search"), calls=api_calls.get("web", 0), hits=api_hits.get("web", 0)),
            _t("word.api_total", n=sum(api_calls.values())),
        ]

        # LLM token usage
        if llm_stats:
            primary = llm_stats.get("primary", {})
            harvest = llm_stats.get("harvest", {})
            params.extend([
                "",
                _t("word.llm_usage_heading"),
                _t("word.primary_usage", model=primary.get("model", "?"), reqs=primary.get("requests", 0), total=f"{primary.get('total_tokens', 0):,}", prompt=f"{primary.get('prompt_tokens', 0):,}", compl=f"{primary.get('completion_tokens', 0):,}"),
            ])
            if harvest.get("requests", 0) > 0:
                params.append(
                    _t("word.harvest_usage", model=harvest.get("model", "?"), reqs=harvest.get("requests", 0), total=f"{harvest.get('total_tokens', 0):,}")
                )
            params.append(
                _t("word.llm_total_requests", n=llm_stats.get('total_requests', 0))
            )
            params.append(
                _t("word.llm_total_tokens", n=f"{llm_stats.get('total_tokens', 0):,}")
            )
    else:
        # Normal research
        params = [
            _t("word.research_id", id=ctx.id),
            _t("word.max_rounds", n=os.environ.get('MAX_RESEARCH_ROUNDS', '5')),
            _t("word.rounds_done", n=ctx.rounds_completed),
            _t("word.sources_total", n=len(ctx.sources)),
            _t("word.extracts_total", n=len(ctx.extracts)),
            _t("word.duration_s", n=f"{ctx.duration_seconds:.0f}"),
        ]

    if ctx.output_schema:
        params.append(_t("word.output_format", v=ctx.output_schema.format_type))
        params.append(_t("word.output_style", v=ctx.output_schema.style))
    for param in params:
        if param.startswith("---"):
            run = p.add_run(f"\n{param.strip('- ')}\n")
            run.font.size = Pt(9)
            run.bold = True
        elif param == "":
            p.add_run("\n")
        else:
            run = p.add_run(f"{param}\n")
            run.font.size = Pt(9)

    # Notice
    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run(
        _t("word.ai_notice")
    )
    run.font.size = Pt(8)
    run.italic = True
    run.font.color.rgb = RGBColor(120, 120, 120)


# ─── Helper functions ─────────────────────────────────────────────

def _format_timestamp(ts: str) -> str:
    """ISO timestamp → readable date and time."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%d.%m.%Y %H:%M:%S")
    except Exception:
        return ts[:19] if len(ts) >= 19 else ts


# ─── Main export function ────────────────────────────────────────

def export_research_to_word(ctx, filename: str | None = None) -> str | None:
    """Export a research report as a Word document.

    Contains:
    - title page with metadata
    - report (Markdown → Word, with hyperlinks)
    - appendix A: list of sources (grouped, with hyperlinks)
    - appendix B: extracts (grouped by question)
    - appendix C: research log (progress)
    - appendix D: pipeline run and metadata (tool URL, LLM, transparency)
    """
    _EXPORT_LANG.set(getattr(ctx, "output_language", None) or "en")
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        logger.warning("python-docx not installed — pip install python-docx")
        return None

    try:
        doc = Document()

        # ── Default font ──
        style = doc.styles["Normal"]
        font = style.font
        font.name = "Calibri"
        font.size = Pt(10)

        # ── Title ──
        title = _t("word.title_default")
        if ctx.output_schema and ctx.output_schema.title:
            title = ctx.output_schema.title

        heading = doc.add_heading(title, level=0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Request
        if ctx.query:
            p_query = doc.add_paragraph()
            p_query.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p_query.add_run(_t("word.request_short", query=ctx.query[:200]))
            run.font.size = Pt(10)
            run.italic = True
            run.font.color.rgb = RGBColor(80, 80, 80)

        # Meta line
        meta = doc.add_paragraph()
        meta.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Languages for the meta line
        lang_names_short = {
            "de": "DE", "en": "EN", "zh": "ZH", "pt": "PT", "es": "ES",
            "fr": "FR", "ja": "JA", "ko": "KO", "ru": "RU",
            "it": "IT", "nl": "NL", "ar": "AR", "pl": "PL", "tr": "TR",
        }
        search_langs = ""
        if ctx.search_stats and ctx.search_stats.get("languages"):
            langs = ctx.search_stats["languages"]
            search_langs = _t("word.meta_languages", langs='/'.join(lang_names_short.get(l, l) for l in langs))

        meta_text = (
            _t("word.meta_line", date=datetime.now().strftime(_t("word.date_format")), sources=len(ctx.sources), extracts=len(ctx.extracts), rounds=ctx.rounds_completed, secs=f"{ctx.duration_seconds:.0f}", langs=search_langs)
        )
        run = meta.add_run(meta_text)
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(128, 128, 128)

        # LLM notice
        model_name = os.environ.get("LLM_MODEL_NAME", "")
        if model_name:
            p_llm = doc.add_paragraph()
            p_llm.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p_llm.add_run(_t("word.language_model", model=model_name))
            run.font.size = Pt(8)
            run.font.color.rgb = RGBColor(160, 160, 160)

        doc.add_paragraph()

        # ── Report ──
        if ctx.final_report:
            _add_markdown_to_doc(doc, ctx.final_report)

        # ── Appendices ──
        is_lit_check = (
            ctx.output_schema
            and ctx.output_schema.format_type == "literature_check"
        )

        # Literature check: sources/extracts/progress are already part of
        # the main report → skip the standard appendices
        if not is_lit_check:
            if ctx.sources:
                doc.add_page_break()
                _add_sources_appendix(doc, ctx.sources)

            if ctx.harvest_results or ctx.extracts:
                doc.add_page_break()
                _add_extracts_appendix(doc, ctx.harvest_results, ctx.extracts)

            # Contradictions between sources (if any)
            contradictions = getattr(ctx, 'contradiction_warnings', None)
            if contradictions:
                doc.add_page_break()
                _add_contradictions_appendix(doc, contradictions)

            if ctx.progress_log:
                doc.add_page_break()
                _add_progress_appendix(doc, ctx.progress_log, ctx)

        # Appendix D: pipeline run — full process transparency.
        # Useful for the literature check too (classifier/filter decisions
        # matter there just as much). Fail-open: an error here must not
        # abort the export.
        try:
            doc.add_page_break()
            _add_pipeline_run_appendix(doc, ctx)
        except Exception as e:
            logger.warning(f"Pipeline-run appendix failed: {e}")

        doc.add_page_break()
        _add_metadata_appendix(doc, ctx)

        # Universal report footer (for all modes: research + analysis)
        try:
            from src.exporters.report_footer import add_word_footer_appendix
            add_word_footer_appendix(doc, ctx)
        except Exception as e:
            logger.warning(f"Footer appendix failed: {e}")

        # ── Footer ──
        for section in doc.sections:
            footer = section.footer
            footer.is_linked_to_previous = False
            p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            from src.institution import operator_line
            _op = operator_line()
            run = p.add_run(f"{TOOL_NAME} \u00b7 {_op}" if _op else TOOL_NAME)
            run.font.size = Pt(7)
            run.font.color.rgb = RGBColor(160, 160, 160)

        # ── Save ──
        if not filename:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(
                c for c in title[:40] if c.isalnum() or c in "-_ "
            ).strip().replace(" ", "_")
            filename = f"research_{safe_title}_{ts}.docx"

        # Writes to GRADIO_TEMP_DIR — Gradio tracks the file and deletes
        # it automatically via delete_cache. NamedTemporaryFile adds a
        # unique suffix so that parallel exports with the same title do
        # not collide.
        gradio_temp = os.environ.get("GRADIO_TEMP_DIR") or tempfile.gettempdir()
        os.makedirs(gradio_temp, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".docx",
            prefix=filename.removesuffix(".docx") + "_",
            delete=False, dir=gradio_temp,
        ) as f:
            filepath = f.name
        doc.save(filepath)

        logger.info(f"Word export: {filepath} ({os.path.getsize(filepath)} bytes)")
        return filepath

    except Exception as e:
        logger.error(f"Word export error: {e}", exc_info=True)
        return None
