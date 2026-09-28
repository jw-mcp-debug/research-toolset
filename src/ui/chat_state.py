"""
ChatState — wrapper for the chat message list of the Gradio UI.

Working directly with `chatbot[-1]`, `chatbot[:-1]` and similar index
operations has two problems:

1. **Prone to IndexError**: `chatbot[-1] = {...}` crashes on an empty
   list, and the defensive guard `if chatbot and chatbot[-1].get("role")
   == "assistant"` ends up duplicated in many places.

2. **Unclear responsibility**: what does `chatbot = list(chatbot[:-1])`
   mean? "Remove the last message — usually to replace it with a new
   one". The pattern recurs with subtly different semantics.

This class wraps the list in a small data type with clearly named
operations. Adoption can be gradual — callers can keep using the
original list or switch to the ChatState methods step by step, because
ChatState works in place on the given list (same identity, no copy).

For Gradio streaming, which expects a new list per yield, there is
`snapshot()` — a shallow copy.

This class is Gradio-AGNOSTIC: no `gr.update` calls, no component use.
It can be unit-tested on its own.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional


# Allowed roles — as in Gradio's Chatbot component
USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"  # rare in the UI, but supported by the schema


class ChatState:
    """Wrapper for a list of chat messages.

    A message is a dict: `{"role": "user"|"assistant"|"system",
    "content": str}`. Other fields (e.g. metadata, name) are passed
    through but not interpreted.

    The class works in place on the given list — mutations are thus
    visible outside as well. That is intentional: gradual adoption in
    the UI code is only possible if the old (`chatbot[-1] = ...`) and
    new (`chat.replace_last_assistant(...)`) styles work on the same
    data structure.

    If the caller needs a real copy (e.g. for Gradio streaming), it
    uses `snapshot()`.
    """

    __slots__ = ("_messages",)

    def __init__(self, messages: Optional[Iterable[dict]] = None):
        if messages is None:
            self._messages: list[dict] = []
        elif isinstance(messages, list):
            # NO copy — we want to work in place
            self._messages = messages
        else:
            self._messages = list(messages)

    # ── Properties ─────────────────────────────────────────────

    @property
    def messages(self) -> list[dict]:
        """Live reference to the message list.

        Caution: external mutations are possible, but should rather go
        through the ChatState methods, otherwise validation is bypassed.
        """
        return self._messages

    def snapshot(self) -> list[dict]:
        """Shallow copy of the messages.

        For Gradio streaming yields, where a new list is expected per
        step. The message dicts themselves are not deep-copied —
        mutations of them are visible in the original list too (no
        problem in practice, because the message dicts are used as
        immutable).
        """
        return list(self._messages)

    # ── Queries ──────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return not self._messages

    @property
    def last_role(self) -> Optional[str]:
        if not self._messages:
            return None
        return self._messages[-1].get("role")

    @property
    def last_content(self) -> str:
        """Content of the last message, or an empty string if there is none."""
        if not self._messages:
            return ""
        return self._messages[-1].get("content", "") or ""

    @property
    def last_user_content(self) -> Optional[str]:
        """Last user text, or None if there is no user message."""
        for msg in reversed(self._messages):
            if msg.get("role") == USER:
                return msg.get("content", "")
        return None

    @property
    def last_assistant_content(self) -> Optional[str]:
        for msg in reversed(self._messages):
            if msg.get("role") == ASSISTANT:
                return msg.get("content", "")
        return None

    def has_pending_assistant(self) -> bool:
        """True if the last message is an assistant message.

        This is the pattern `chatbot and chatbot[-1].get("role") ==
        "assistant"` in one place.
        """
        return self.last_role == ASSISTANT

    def has_pending_user(self) -> bool:
        return self.last_role == USER

    # ── Mutations ────────────────────────────────────────────────

    def append_user(self, content: str) -> None:
        """Append a user message at the end."""
        self._messages.append({"role": USER, "content": content})

    def append_assistant(self, content: str) -> None:
        """Append an assistant message at the end."""
        self._messages.append({"role": ASSISTANT, "content": content})

    def replace_last_assistant(
        self,
        content: str,
        *,
        append_if_missing: bool = True,
    ) -> None:
        """Replace the content of the last assistant message.

        The pattern:
            if chatbot and chatbot[-1].get("role") == "assistant":
                chatbot[-1] = {"role": "assistant", "content": new_content}

        in one method:
            chat.replace_last_assistant(new_content)

        Args:
            content: the new text.
            append_if_missing: if the last message is NOT an assistant
                message, `True` (default) appends a new assistant
                message. With `False` the method is a no-op (safe
                behaviour — no user message is overwritten).
        """
        if self.has_pending_assistant():
            # Mutate the dict fields in place instead of replacing the dict —
            # so any additional fields (e.g. metadata) survive.
            self._messages[-1]["content"] = content
        elif append_if_missing:
            self.append_assistant(content)
        # else: No-op

    def drop_last(self, n: int = 1) -> None:
        """Remove the last n messages.

        Replaces the pattern `chatbot = list(chatbot[:-1])`.
        """
        if n <= 0:
            return
        if n >= len(self._messages):
            self._messages.clear()
        else:
            del self._messages[-n:]

    def clear(self) -> None:
        self._messages.clear()

    # ── Container protocol ───────────────────────────────────────
    # Makes ChatState a drop-in for `len(chatbot)`, `for msg in chatbot`,
    # etc. — so callers can switch gradually without changing every
    # iteration site at once.

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._messages)

    def __bool__(self) -> bool:
        return bool(self._messages)

    def __getitem__(self, index):
        return self._messages[index]

    def __repr__(self) -> str:
        n = len(self._messages)
        last = self.last_role or "—"
        return f"ChatState(messages={n}, last_role={last!r})"
