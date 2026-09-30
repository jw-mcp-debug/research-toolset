"""Clickable action phrases in chat answers.

The chat system prompt tells the model to write these phrases; a small
script in the UI turns them into clickable links. Both sides are built
from this table, so the prompt text and the recognition pattern cannot
drift apart. Each action has one primary phrase (used in the prompt)
and aliases the model may produce instead, e.g. when it answers in the
user's language.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


def _js_escape(text: str) -> str:
    """Escape only JavaScript regex metacharacters (valid in any flag mode)."""
    return re.sub(r"([.*+?^${}()|\[\]\\/])", r"\\\1", text)


@dataclass(frozen=True)
class ChatAction:
    action: str          # data-action value handled by the UI script
    emoji: str
    phrase: str          # primary phrase, used in the system prompt
    aliases: tuple[str, ...]
    title: str           # tooltip

    @property
    def label(self) -> str:
        return f"{self.emoji} {self.phrase}"

    def js_regex(self) -> str:
        """JavaScript regex source matching emoji + any known phrase."""
        alts = "|".join(_js_escape(p) for p in (self.phrase, *self.aliases))
        return f"({_js_escape(self.emoji)}\\s*(?:{alts}))"


CHAT_ACTIONS: tuple[ChatAction, ...] = (
    ChatAction("focus", "💬", "Auftrag besprechen",
               ("Discuss request", "Discuss further", "Weiter diskutieren", "Besprechen"),
               "Zum Eingabefeld springen"),
    ChatAction("adopt", "📋", "Vorschlag übernehmen",
               ("Adopt suggestion", "Use as request", "Als Auftrag nutzen", "Übernehmen"),
               "Bereinigten Vorschlag ins Eingabefeld übernehmen"),
    ChatAction("research", "🔍", "Recherche starten",
               ("Start research", "Research now", "Jetzt recherchieren"),
               "Recherche starten"),
    ChatAction("litcheck", "📚", "Literatur prüfen",
               ("Check references", "Check bibliography", "Check literature", "Literaturprüfung"),
               "Literaturverzeichnis prüfen"),
)

ACTIONS_BY_NAME = {a.action: a for a in CHAT_ACTIONS}


def js_replace_rules() -> str:
    """JavaScript statements that wrap every action phrase in a link span."""
    out = []
    for a in CHAT_ACTIONS:
        out.append(
            "changed = changed.replace(new RegExp(%s, 'g'), "
            "'<span class=\"chat-action\" data-action=\"%s\" "
            "style=\"cursor:pointer;color:var(--primary-500);"
            "text-decoration:underline;font-weight:600\" title=%s>$1</span>');"
            % (json.dumps(a.js_regex(), ensure_ascii=False), a.action,
               json.dumps(a.title, ensure_ascii=False).replace("'", "\\'"))
        )
    return "\n                ".join(out)
