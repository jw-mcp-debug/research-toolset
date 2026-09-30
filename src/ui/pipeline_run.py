"""
Pipeline run tab — rendering layer.

Contract:
    render_pipeline_run(ctx) -> str   (Markdown)

UI-agnostic: takes a HarvestContext (or any object with the same
attributes) and returns Markdown. The Gradio wiring lives in
`gradio_app.py` (reactive `app_state.change()` handler).

Robustness: every section is encapsulated on its own. A missing or
broken sub-structure must never empty the whole tab — when in doubt the
section is skipped, not raised.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─── small helpers ───────────────────────────────────────────────────


def _g(obj: Any, name: str, default: Any = None) -> Any:
    """getattr with a default — tolerates None and missing attributes."""
    if obj is None:
        return default
    return getattr(obj, name, default)


def _short(text: Any, n: int = 300) -> str:
    s = str(text or "").strip()
    return s if len(s) <= n else s[:n] + " …"


def _bool_icon(v: Any) -> str:
    return "✅" if v else "❌"


def _section(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"## {title}\n\n{body}\n"


# ─── Section renderers (each defensive) ──────────────────────────────


def _render_header(ctx: Any) -> str:
    status = _g(ctx, "status", "?")
    rounds = _g(ctx, "rounds_completed", 0)
    n_src = len(_g(ctx, "sources", []) or [])
    n_ext = len(_g(ctx, "extracts", []) or [])
    try:
        dur = f"{ctx.duration_seconds:.0f}s"
    except Exception:
        dur = "?"
    lines = [
        f"**Status:** `{status}`  |  **Runden:** {rounds}  |  "
        f"**Quellen:** {n_src}  |  **Extrakte:** {n_ext}  |  "
        f"**Dauer:** {dur}",
    ]
    q = _short(_g(ctx, "query", ""), 400)
    if q:
        lines.append(f"\n> {q}")
    err = _g(ctx, "error_message", "")
    if err:
        lines.append(f"\n⚠️ **Fehler:** {_short(err, 500)}")
    return _section("🧠 Pipeline-Lauf", "\n".join(lines))


def _render_output_schema(ctx: Any) -> str:
    schema = _g(ctx, "output_schema")
    if not schema:
        return ""
    rows = [
        f"- **Titel:** {_g(schema, 'title', '—')}",
        f"- **Format:** `{_g(schema, 'format_type', '—')}`  |  "
        f"**Sprache:** `{_g(schema, 'language', '—')}`",
    ]
    secs = _g(schema, "sections", []) or []
    if secs:
        names = []
        for s in secs:
            if isinstance(s, dict):
                names.append(str(s.get("title") or s.get("name") or s))
            else:
                names.append(str(s))
        rows.append(f"- **Abschnitte:** {', '.join(names)}")
    guidance = _short(_g(schema, "synthesis_guidance", ""), 400)
    if guidance:
        rows.append(f"- **Vorgaben für die Synthese:** {guidance}")
    return _section("📐 Ausgabeschema", "\n".join(rows))


def _render_plan(ctx: Any) -> str:
    plan = _g(ctx, "research_plan")
    if not plan:
        return ""
    parts: list[str] = []
    summary = _short(_g(plan, "summary", ""), 600)
    if summary:
        parts.append(summary + "\n")

    questions = _g(plan, "questions", []) or []
    for q in questions:
        qid = _g(q, "id", "?")
        qtext = _g(q, "question", "")
        prio = _g(q, "priority", "—")
        scope = _g(q, "source_scope", "—")
        langs = ", ".join(_g(q, "search_langs", []) or [])
        answered = "beantwortet" if _g(q, "answered", False) else "offen"
        parts.append(
            f"**[{qid}]** {qtext}\n"
            f"  · Priorität `{prio}` · Quellenbereich `{scope}` · "
            f"Sprachen `{langs or '—'}` · {answered}"
        )
        terms = _g(q, "search_terms", []) or []
        if terms:
            preview = ", ".join(f"`{t}`" for t in terms[:8])
            if len(terms) > 8:
                preview += f" (+{len(terms) - 8})"
            parts.append(f"  · Suchbegriffe: {preview}")

    durls = _g(plan, "direct_urls", []) or []
    if durls:
        parts.append(f"\n**Direkte URLs ({len(durls)}):**")
        for d in durls[:15]:
            parts.append(f"  · {_g(d, 'url', '')} — {_short(_g(d, 'reason', ''), 120)}")

    repos = _g(plan, "git_repos", []) or []
    if repos:
        parts.append(f"\n**Git-Repos ({len(repos)}):**")
        for r in repos[:15]:
            parts.append(
                f"  · {_g(r, 'owner', '')}/{_g(r, 'repo', '')} "
                f"({_g(r, 'platform', 'github')})"
            )

    zqs = _g(plan, "directory_queries", []) or []
    if zqs:
        parts.append(f"\n**Abfragen im Personenverzeichnis:** {len(zqs)}")

    fqs = _g(plan, "followup_queries", []) or []
    if fqs:
        parts.append(f"\n**Folgeanfragen (aus der Abdeckungsprüfung):** {len(fqs)}")

    return _section("🗺️ Rechercheplan", "\n".join(parts))


def _render_query_anchor(ctx: Any) -> str:
    anchor = _g(ctx, "query_anchor")
    if not anchor:
        return ""
    typ = _g(anchor, "type", "?")
    target = _g(anchor, "target", "")
    conf = _g(anchor, "confidence", 0.0)
    fb = _g(anchor, "fallback_used", False)
    reasoning = _short(_g(anchor, "reasoning", ""), 400)
    try:
        is_person = bool(anchor.is_person())
    except Exception:
        is_person = False
    body = (
        f"- **Typ:** `{typ}`  |  **Ziel:** {target or '—'}  |  "
        f"**Konfidenz:** {conf:.2f}  |  Fallback: {_bool_icon(fb)}\n"
        f"- **Personen-Halluzinationsfilter aktiv:** "
        f"{_bool_icon(is_person)}"
    )
    if reasoning:
        body += f"\n- **Begründung:** {reasoning}"
    return _section("⚓ Query-Anker", body)


def _render_search_log(ctx: Any) -> str:
    stats = _g(ctx, "search_stats", {}) or {}
    if not stats:
        return ""
    rows = []
    langs = stats.get("languages") or []
    if langs:
        rows.append(f"- **Suchsprachen:** {', '.join(langs)}")
    tbl = stats.get("terms_by_lang") or {}
    for lang, terms in tbl.items():
        terms = terms or []
        preview = ", ".join(f"`{t}`" for t in terms[:6])
        if len(terms) > 6:
            preview += f" (+{len(terms) - 6})"
        rows.append(f"  · **{lang}:** {preview}")
    for k, v in stats.items():
        if k in ("languages", "terms_by_lang"):
            continue
        rows.append(f"- **{k}:** {_short(v, 200)}")
    return _section("🔎 Suchverlauf", "\n".join(rows))


def _render_filter_stats(ctx: Any) -> str:
    rounds = _g(ctx, "filter_stats_per_round", []) or []
    if not rounds:
        return ""
    parts = []
    for i, rnd in enumerate(rounds, 1):
        if not isinstance(rnd, dict) or not rnd:
            continue
        parts.append(f"**Runde {i}:**")
        for fname, fstats in rnd.items():
            if isinstance(fstats, dict):
                act = fstats.get("activated")
                rej = fstats.get("rejected", 0)
                kept = fstats.get("kept", fstats.get("passed", "—"))
                parts.append(
                    f"  · `{fname}` — aktiv {_bool_icon(act)}, "
                    f"verworfen {rej}, behalten {kept}"
                )
            else:
                parts.append(f"  · `{fname}`: {_short(fstats, 120)}")
    return _section("🧪 Filterstatistik", "\n".join(parts))


def _render_coverage(ctx: Any) -> str:
    per_round = _g(ctx, "coverage_per_round", []) or []
    if not per_round:
        return ""
    parts = []
    for i, rnd in enumerate(per_round, 1):
        if not isinstance(rnd, dict) or not rnd:
            continue
        parts.append(f"**Runde {i}:**")
        for qid, cov in rnd.items():
            if isinstance(cov, dict):
                c = cov.get("coverage", "?")
                conf = cov.get("confidence", 0.0)
                miss = cov.get("missing_aspects") or []
                line = f"  · **{qid}:** `{c}` (Konf. {conf:.2f})"
                if miss:
                    line += f" — fehlt: {', '.join(str(m) for m in miss[:4])}"
                parts.append(line)
            else:
                parts.append(f"  · **{qid}:** {_short(cov, 120)}")
    return _section("📊 Abdeckung je Frage", "\n".join(parts))


def _render_map_answers(ctx: Any) -> str:
    answers = _g(ctx, "map_answers", []) or []
    if not answers:
        return ""
    parts = []
    for a in answers:
        if not isinstance(a, dict):
            continue
        qid = a.get("question_id", "?")
        q = a.get("question", "")
        n = a.get("n_extracts", 0)
        ans = _short(a.get("answer", ""), 800)
        parts.append(f"**[{qid}]** {q}  _(aus {n} Extrakten)_\n\n{ans}\n")
    return _section("🧩 Synthese-Map (Antworten je Frage)", "\n".join(parts))


def _render_diagnosis(ctx: Any) -> str:
    diag = _g(ctx, "final_diagnosis")
    if not diag:
        return ""
    if isinstance(diag, dict):
        body = "\n".join(
            f"- **{k}:** {_short(v, 300)}" for k, v in diag.items()
        )
    else:
        body = _short(diag, 600)
    return _section("🩺 Diagnose", body)


def _render_quality_fulfillment(ctx: Any) -> str:
    rq = _g(ctx, "report_quality")
    qf = _g(ctx, "query_fulfillment")
    if not rq and not qf:
        return ""
    parts = []
    if isinstance(rq, dict):
        parts.append(
            f"**Berichtsqualität:** bestanden {_bool_icon(rq.get('passed'))}"
            f" · {_short(rq.get('rating', ''), 300)}"
        )
        issues = rq.get("issues") or []
        for m in issues[:10]:
            if isinstance(m, dict):
                parts.append(
                    f"  · {_short(m.get('description') or m.get('typ'), 200)}"
                )
            else:
                parts.append(f"  · {_short(m, 200)}")
        if rq.get("fallback_used"):
            parts.append("  · _(Fallback-Bewertung verwendet)_")
    if isinstance(qf, dict):
        parts.append(
            f"\n**Erfüllung der Anfrage:** erfüllt "
            f"{_bool_icon(qf.get('fulfilled'))}"
            f" · {_short(qf.get('assessment', ''), 300)}"
        )
        if qf.get("rework"):
            parts.append(f"  · Nacharbeit: {_short(qf.get('rework'), 300)}")
        if qf.get("fallback_used"):
            parts.append("  · _(Fallback-Bewertung verwendet)_")
    return _section("✅ Qualität & Erfüllung", "\n".join(parts))


def _render_report_revision(ctx: Any) -> str:
    rev = _g(ctx, "report_revision")
    if not rev or not isinstance(rev, dict):
        return ""
    if not rev.get("applied"):
        reason = _short(rev.get("reason", ""), 300)
        return _section(
            "✏️ Berichtsrevision",
            f"Nicht angewendet — {reason or 'keine Widersprüche mit hoher Konfidenz'}",
        )
    n = rev.get("n_corrected", 0)
    facts = rev.get("factoids_corrected") or []
    body = [
        f"Angewendet — **{n}** Aussage(n) korrigiert "
        f"(Länge {rev.get('original_length', '?')} → "
        f"{rev.get('corrected_length', '?')} Zeichen)",
    ]
    for f in facts[:10]:
        body.append(f"  · {_short(f, 200)}")
    return _section("✏️ Berichtsrevision", "\n".join(body))


def _render_factoids(ctx: Any) -> str:
    facts = _g(ctx, "factoid_verifications", []) or []
    if not facts:
        return ""
    parts = []
    for f in facts[:40]:
        if not isinstance(f, dict):
            continue
        v = f.get("verified", "?")
        conf = f.get("confidence", 0.0)
        icon = {
            "supported": "✅", "contradicted": "❌",
            "unverifiable": "❓", "unsupported": "⚠️",
        }.get(str(v), "·")
        parts.append(
            f"{icon} `{v}` (Konf. {conf:.2f}) — "
            f"{_short(f.get('factoid', ''), 220)}"
        )
    return _section("🔬 Faktoid-Verifikation", "\n".join(parts))


def _render_classifier_calls(ctx: Any) -> str:
    calls = _g(ctx, "classifier_calls", []) or []
    if not calls:
        return ""
    parts = []
    for c in calls:
        # Accepts ClassifierCall objects or plain dicts
        if hasattr(c, "to_dict"):
            try:
                c = c.to_dict()
            except Exception:
                pass
        if isinstance(c, dict):
            name = c.get("name", "?")
            conf = c.get("confidence", 0.0)
            fb = c.get("fallback_used", False)
            dur = c.get("duration_seconds", 0.0)
            line = (
                f"- `{name}` — Konf. {conf:.2f}, {dur:.2f}s"
                f"{', Fallback ⚠️' if fb else ''}"
            )
            if fb and c.get("fallback_reason"):
                line += f" ({_short(c.get('fallback_reason'), 120)})"
            parts.append(line)
        else:
            parts.append(f"- {_short(c, 160)}")
    head = f"_{len(calls)} Klassifikator-Aufruf(e)_\n\n"
    return _section("🧮 Klassifikator-Aufrufe", head + "\n".join(parts))


# ─── public entry point ───────────────────────────────────────────


_RENDERERS = (
    _render_header,
    _render_output_schema,
    _render_plan,
    _render_query_anchor,
    _render_search_log,
    _render_coverage,
    _render_filter_stats,
    _render_map_answers,
    _render_diagnosis,
    _render_quality_fulfillment,
    _render_report_revision,
    _render_factoids,
    _render_classifier_calls,
)


def render_pipeline_run(ctx: Any) -> str:
    """Render a HarvestContext as pipeline-run Markdown.

    Defensive: every section is isolated. A section that raises is
    skipped (with a log warning), the rest stays visible.
    """
    if ctx is None:
        return "*Noch keine Recherche gestartet.*"

    blocks: list[str] = []
    for fn in _RENDERERS:
        try:
            out = fn(ctx)
            if out:
                blocks.append(out)
        except Exception as e:  # a section must never empty the tab
            logger.warning(
                "Pipeline run: section %s failed: %s",
                getattr(fn, "__name__", fn), e,
            )

    if not blocks:
        return "*Recherche läuft — noch keine Zwischenergebnisse.*"
    return "\n---\n\n".join(blocks)
