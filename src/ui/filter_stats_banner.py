"""
Filter statistics banner — makes filter losses visible in the UI.

Filters can discard hundreds of extracts without the user noticing; the
final report is then empty and nothing explains why. This function
formats the filter statistics from `HarvestContext` into a Markdown
banner that the UI can show. It is **Gradio-agnostic** — it only
returns a Markdown string. Wiring it to the Gradio component is UI code
in `gradio_app.py`.

What the banner contains:
  - number of sources per round, number of extracts
  - per filter: rejected/total, loss rate
  - if a classifier diagnosis exists: diagnosis + recommendation
  - coverage status per question (answered/partial/unanswered/filter_blocked)
  - optional: quality check and fulfilment results

Deliberately no gr.update, no gr.HTML, no Gradio import — only a
Markdown string, so the formatting can be unit-tested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.models import HarvestContext


# Thresholds above which a filter is marked as "conspicuous" in the banner
HIGH_LOSS_RATE = 0.5    # > 50 % loss → warning marker in the banner
CRITICAL_LOSS_RATE = 0.9  # > 90 % loss → critical marker (typical of a wrongly activated filter)


def format_filter_stats_banner(ctx: "HarvestContext") -> str:
    """Format a Markdown banner with filter statistics and diagnosis.

    For a clean run (no conspicuous filters, no problems): a short
    banner with the main figures.

    For filter problems or a problematic diagnosis: a detailed banner
    with recommendations.

    Args:
        ctx: the research context after the pipeline run.

    Returns:
        Markdown text. Empty string if ctx has no relevant data
        (e.g. a pipeline aborted before producing statistics).
    """
    sections: list[str] = []

    # ── Header ──
    n_sources = len(getattr(ctx, "sources", []) or [])
    n_extracts = len(getattr(ctx, "extracts", []) or [])
    n_rounds = getattr(ctx, "rounds_completed", 0) or 0
    if n_sources == 0 and n_extracts == 0:
        # Probably aborted before any activity — no banner
        return ""

    sections.append(
        f"### 📊 Recherchestatistik\n\n"
        f"- **Runden:** {n_rounds}\n"
        f"- **Abgerufene Quellen:** {n_sources}\n"
        f"- **Gewonnene Extrakte:** {n_extracts}"
    )

    # ── Filter losses ──
    filter_section = _format_filter_losses(ctx)
    if filter_section:
        sections.append(filter_section)

    # ── Coverage per question ──
    coverage_section = _format_coverage(ctx)
    if coverage_section:
        sections.append(coverage_section)

    # ── Diagnose ──
    diag_section = _format_diagnosis(ctx)
    if diag_section:
        sections.append(diag_section)

    # ── Quality + Fulfillment ──
    quality_section = _format_quality_and_fulfillment(ctx)
    if quality_section:
        sections.append(quality_section)

    return "\n\n".join(sections)


# ── Filter losses ───────────────────────────────────────────────


def _format_filter_losses(ctx: "HarvestContext") -> str:
    """Per filter: rejected/total + loss rate, with a marker when conspicuous."""
    rounds = getattr(ctx, "filter_stats_per_round", None) or []
    if not rounds:
        return ""

    # Aggregate over all rounds
    aggregate: dict[str, dict] = {}
    for round_stats in rounds:
        for filter_name, stats in (round_stats or {}).items():
            agg = aggregate.setdefault(filter_name, {
                "rejected": 0, "total": 0, "activated_in_rounds": 0,
            })
            agg["rejected"] += stats.get("rejected", 0)
            agg["total"] += stats.get("total", 0)
            if stats.get("activated"):
                agg["activated_in_rounds"] += 1

    if not aggregate:
        return ""

    # Only show filters that were activated at least once
    active = {
        name: agg for name, agg in aggregate.items()
        if agg["activated_in_rounds"] > 0 and agg["total"] > 0
    }
    if not active:
        return "### 🛡️ Filter\n\n_Keine Filter aktiv._"

    lines = ["### 🛡️ Filterstatistik\n"]
    for name, agg in sorted(active.items()):
        rejected = agg["rejected"]
        total = agg["total"]
        rate = rejected / total if total else 0.0
        marker = ""
        if rate >= CRITICAL_LOSS_RATE:
            marker = " ⚠️ **KRITISCH**"
        elif rate >= HIGH_LOSS_RATE:
            marker = " ⚠️"
        lines.append(
            f"- **{name}:** {rejected}/{total} verworfen "
            f"({rate * 100:.0f}%){marker}"
        )

    # If a filter loses critically much: append a notice block
    critical = [n for n, a in active.items()
                if a["total"] > 0 and a["rejected"] / a["total"] >= CRITICAL_LOSS_RATE]
    if critical:
        lines.append(
            "\n> **Hinweis:** Filter mit mehr als 90 % Verlust haben fast alle "
            "Extrakte verworfen. Wirkt der Bericht trotzdem leer, kann eine "
            "Recherche ohne diesen Filter andere Ergebnisse liefern.\n"
            f"> Betroffen: `{', '.join(sorted(critical))}`"
        )
    return "\n".join(lines)


# ── Coverage ──────────────────────────────────────────────────────


def _format_coverage(ctx: "HarvestContext") -> str:
    """Coverage status per question (last round)."""
    cov_per_round = getattr(ctx, "coverage_per_round", None) or []
    if not cov_per_round:
        return ""
    last = (cov_per_round[-1] or {}).get("results", {})
    if not last:
        return ""

    # Counter per status
    counts = {"answered": 0, "partial": 0,
              "unanswered": 0, "filter_blocked": 0}
    for c in last.values():
        cov = c.get("coverage", "partial")
        if cov in counts:
            counts[cov] += 1

    # Markdown
    icon_map = {
        "answered": "✅", "partial": "🟡",
        "unanswered": "⛔", "filter_blocked": "🛡️",
    }
    status_labels = {
        "answered": "beantwortet", "partial": "teilweise",
        "unanswered": "unbeantwortet", "filter_blocked": "vom Filter blockiert",
    }
    summary_parts = []
    for status, n in counts.items():
        if n > 0:
            summary_parts.append(f"{icon_map[status]} {n} {status_labels[status]}")

    if not summary_parts:
        return ""

    lines = ["### 🎯 Abdeckung je Frage\n"]
    lines.append(" · ".join(summary_parts))

    # Details for unanswered or filter_blocked questions
    issues = [
        (qid, c) for qid, c in last.items()
        if c.get("coverage") in ("unanswered", "filter_blocked")
    ]
    if issues:
        lines.append("")
        for qid, c in issues:
            cov = c.get("coverage", "")
            reason = (c.get("reasoning") or "").strip()
            if len(reason) > 200:
                reason = reason[:200] + "…"
            lines.append(
                f"- **{qid}** ({cov}){': ' + reason if reason else ''}"
            )
    return "\n".join(lines)


# ── Diagnose ─────────────────────────────────────────────────────


def _format_diagnosis(ctx: "HarvestContext") -> str:
    diag = getattr(ctx, "final_diagnosis", None)
    if not diag:
        return ""

    diagnosis = diag.get("diagnosis", "")
    if not diagnosis:
        return ""

    # With "successful", a minimal banner — anything more repeats the header
    if diag.get("is_successful"):
        return ""

    icon = "⚠️" if diag.get("is_problematic") else "ℹ️"
    lines = [f"### {icon} Diagnose"]
    msg = (diag.get("user_message") or "").strip()
    if msg:
        lines.append(f"\n{msg}")
    rem = (diag.get("remediation") or "").strip()
    if rem:
        lines.append(f"\n**Empfehlung:** {rem}")
    return "\n".join(lines)


# ── Quality + Fulfillment ─────────────────────────────────────────


def _format_quality_and_fulfillment(ctx: "HarvestContext") -> str:
    """Show report quality and fulfilment together."""
    quality = getattr(ctx, "report_quality", None)
    fulfillment = getattr(ctx, "query_fulfillment", None)

    parts = []
    if quality and quality.get("issues"):
        n_issues = len(quality["issues"])
        assessment = quality.get("rating", "")
        if not quality.get("passed"):
            parts.append(
                f"### 📝 Berichtsqualität\n\n"
                f"⚠️ Bewertung: **{assessment}**, {n_issues} Befunde:\n"
            )
            for m in quality["issues"][:5]:  # show at most 5
                art = m.get("art", "?")
                besch = m.get("description", "")
                parts.append(f"- _{art}:_ {besch}")
            if n_issues > 5:
                parts.append(f"- _({n_issues - 5} weitere)_")

    if fulfillment and not fulfillment.get("fulfilled"):
        assessment = fulfillment.get("assessment", "")
        rework = fulfillment.get("rework", "")
        block = ["### 🎯 Erfüllung der Anfrage\n",
                 f"⚠️ Anfrage nicht vollständig erfüllt: {assessment}"]
        if rework:
            block.append(f"\n**Vorschlag:** {rework}")
        parts.append("\n".join(block))

    return "\n\n".join(parts)
