"""
Research modes: SINGLE SOURCE OF TRUTH.

Every mode is defined here EXACTLY ONCE as (ID, dropdown label, route).
Derived from this table:

  - RESEARCH_MODE_ORDER      — order of the dropdown entries (IDs)
  - DEFAULT_RESEARCH_MODE    — preselected mode
  - ANALYSIS_MODE_MAP        — ID → use_case (analysis modes only)
  - ANALYSIS_USE_CASE_ORDER  — order of the analysis use cases
  - resolve_research_route() — exact ID → route resolution
  - mode_choices()           — (label, ID) pairs for the dropdown

`route` is a tuple (kind, use_case):
  ("web", None)              → web research
  ("institution", None)      → institution research (only with a profile)
  ("literature_check", None) → bibliography check
  ("explainer", None)        → in-depth explanation
  ("analysis", "<uc_name>")  → generic analysis use case

Routing is an exact lookup by ID, never a substring match on labels:
several labels share substrings (three modes contain "Literature"), and
labels may be changed or translated without affecting behaviour.
tests/test_research_modes.py guards this.

This module is deliberately Gradio-free and therefore unit-testable
without UI dependencies.
"""

from __future__ import annotations

from typing import Optional

# Route kind: use_case is only set for "analysis".
Route = tuple[str, Optional[str]]

# Stable mode IDs are what the UI stores and routes on; labels are for
# display only and may be translated freely.
# IMPORTANT: the order here = display order in the dropdown.
RESEARCH_MODES: list[tuple[str, str, Route]] = [
    ("web",               "🌐 Webrecherche",        ("web", None)),
    ("institution",       "🏛️ {institution}-Recherche", ("institution", None)),
    ("literature_check",  "📚 Literaturprüfung",     ("literature_check", None)),
    ("literature_finder", "📑 Literatursuche",     ("analysis", "literature_finder")),
    ("explainer",         "📖 Vertiefte Erklärung",      ("explainer", None)),
    ("peer_review",       "🔍 Peer Review",          ("analysis", "peer_review")),
    ("decision_analysis", "⚖️ Entscheidungsanalyse", ("analysis", "decision_analysis")),
    ("research_design",   "🔬 Forschungsdesign",     ("analysis", "research_design")),
    ("grant_proposal",    "💰 Drittmittelantrag",    ("analysis", "grant_proposal")),
    ("literature_review", "📚 Literature Review",    ("analysis", "literature_review")),
]

# ── Derived structures (do NOT maintain by hand) ──
RESEARCH_MODE_ORDER: list[str] = [mode_id for mode_id, _, _ in RESEARCH_MODES]
DEFAULT_RESEARCH_MODE: str = RESEARCH_MODE_ORDER[0]
_MODE_ROUTING: dict[str, Route] = {mode_id: route for mode_id, _, route in RESEARCH_MODES}

#: Mapping mode ID → use_case name (only the generic analysis modes).
ANALYSIS_MODE_MAP: dict[str, str] = {
    mode_id: uc
    for mode_id, _, (kind, uc) in RESEARCH_MODES
    if kind == "analysis" and uc is not None
}

#: Order in which the components lie in `analysis_components_flat`.
ANALYSIS_USE_CASE_ORDER: list[str] = [
    uc for _, _, (kind, uc) in RESEARCH_MODES if kind == "analysis" and uc is not None
]


def resolve_research_route(mode_id: str) -> Route:
    """Resolve a mode ID EXACTLY to its route.

    Unknown or empty IDs fall back to web research.
    """
    return _MODE_ROUTING.get(mode_id or "", ("web", None))


def mode_choices(profile=None) -> list[tuple[str, str]]:
    """(label, id) pairs for the mode dropdown.

    The institution mode is offered only when an institution profile is
    configured; its label carries the institution's short name.
    """
    if profile is None:
        from src.institution import get_profile
        profile = get_profile()
    out = []
    for mode_id, label, _ in RESEARCH_MODES:
        if mode_id == "institution":
            if not profile.configured:
                continue
            label = label.replace("{institution}", profile.label)
        out.append((label, mode_id))
    return out


# ── Two-level selector: research forms + "Analyse …" ──
# The toolbar shows the research forms directly and bundles the analysis
# modes behind one entry, with a second dropdown for the concrete mode.
# The effective mode is still a single ID from RESEARCH_MODES.

#: Value of the "Analyse …" entry in the first dropdown (not a mode ID).
ANALYSIS_GROUP: str = "analysis"
ANALYSIS_GROUP_LABEL: str = "🧠 Analyse …"

#: Modes offered under "Analyse …" (display order follows RESEARCH_MODES).
ANALYSIS_GROUP_MEMBERS: frozenset[str] = frozenset({
    "explainer", "peer_review", "decision_analysis",
    "research_design", "grant_proposal", "literature_review",
})

DEFAULT_ANALYSIS_MODE: str = next(
    m for m in RESEARCH_MODE_ORDER if m in ANALYSIS_GROUP_MEMBERS
)

#: One-line description per mode, shown below the toolbar.
MODE_DESCRIPTIONS: dict[str, str] = {
    "web": "Allgemeine Onlinesuche mit Bericht und Quellenangaben – für "
           "aktuelle Themen, Marktinformationen und Allgemeinwissen.",
    "institution": "Sucht in den Quellen von {institution} (Website, "
                   "Personenverzeichnis) – für Fragen rund um die Einrichtung.",
    "literature_check": "Prüft ein vorhandenes Literaturverzeichnis Eintrag für "
                        "Eintrag gegen CrossRef und OpenAlex, findet Fehler und "
                        "ergänzt DOIs.",
    "literature_finder": "Durchsucht OpenAlex, Semantic Scholar und arXiv zu "
                         "Ihrer Forschungsfrage und bewertet die Treffer.",
    "explainer": "Erklärt ein Thema strukturiert für eine bestimmte Zielgruppe.",
    "peer_review": "Erstellt ein strukturiertes Gutachten zu einem Paper, "
                   "Aspekt für Aspekt.",
    "decision_analysis": "Vergleicht 2–8 Optionen anhand Ihrer Kriterien.",
    "research_design": "Entwickelt aus einer Forschungsfrage Forschungslücke, "
                       "Hypothesen, Methode und Limitationen.",
    "grant_proposal": "Antragsentwurf mit Literaturrecherche zum "
                      "Forschungsstand, Arbeitsplan und Kohärenzprüfung.",
    "literature_review": "Literaturübersicht mit Citation Chasing: je "
                         "Leitfrage eine Synthese, dazu eine Metasynthese.",
}


def mode_group_choices(profile=None) -> list[tuple[str, str]]:
    """(label, value) pairs for the first dropdown: research forms + "Analyse …"."""
    out = [c for c in mode_choices(profile) if c[1] not in ANALYSIS_GROUP_MEMBERS]
    out.append((ANALYSIS_GROUP_LABEL, ANALYSIS_GROUP))
    return out


def analysis_mode_choices(profile=None) -> list[tuple[str, str]]:
    """(label, ID) pairs for the second dropdown (analysis modes only)."""
    return [c for c in mode_choices(profile) if c[1] in ANALYSIS_GROUP_MEMBERS]


def compose_mode(group: str, analysis: str) -> str:
    """Effective mode ID from the two dropdown values."""
    if group == ANALYSIS_GROUP:
        return analysis if analysis in ANALYSIS_GROUP_MEMBERS else DEFAULT_ANALYSIS_MODE
    if group in _MODE_ROUTING and group not in ANALYSIS_GROUP_MEMBERS:
        return group
    return DEFAULT_RESEARCH_MODE


def mode_description(mode_id: str, profile=None) -> str:
    """Short description of a mode ('' if none)."""
    text = MODE_DESCRIPTIONS.get(mode_id or "", "")
    if "{institution}" in text:
        if profile is None:
            from src.institution import get_profile
            profile = get_profile()
        text = text.replace("{institution}", profile.name or profile.label)
    return text
