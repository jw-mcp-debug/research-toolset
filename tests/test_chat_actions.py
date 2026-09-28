"""Chat action phrases: prompt text and UI recognition come from one table."""

import re

from src.ui.chat_actions import CHAT_ACTIONS, js_replace_rules


def test_every_action_has_rule_and_unique_name():
    names = [a.action for a in CHAT_ACTIONS]
    assert len(names) == len(set(names))
    rules = js_replace_rules()
    for a in CHAT_ACTIONS:
        assert f'data-action=\\"{a.action}\\"' in rules or f'data-action="{a.action}"' in rules


def test_regex_matches_primary_phrase_and_aliases():
    for a in CHAT_ACTIONS:
        rx = re.compile(a.js_regex())      # JS subset is valid Python regex here
        assert rx.search(f"text {a.label} text")
        for alias in a.aliases:
            assert rx.search(f"{a.emoji} {alias}")
        assert not rx.search(a.phrase)     # emoji is required
