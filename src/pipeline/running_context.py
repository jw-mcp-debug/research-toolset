"""
RunningContext: accumulated context for long output documents.

Used in use cases such as the in-depth explanation and the grant
proposal, so that later sub-tasks know what earlier sections already
explained — avoids repetition and allows cross-references.

Strategy:
- every section is added as (id, summary)
- on retrieval, the list of summaries is returned
- if the total length exceeds the token limit, older sections are
  summarised or shortened (not removed)
"""

from dataclasses import dataclass
from typing import Optional


# Rough token estimate: 1 token ≈ 4 characters (German/English)
CHARS_PER_TOKEN = 4


@dataclass
class _Section:
    """Internal representation of a section."""
    id: str
    summary: str
    full_text: str = ""           # optional: full text for re-summarisation

    @property
    def char_count(self) -> int:
        return len(self.summary)

    @property
    def estimated_tokens(self) -> int:
        return self.char_count // CHARS_PER_TOKEN


class RunningContext:
    """Accumulated context for long documents.

    Typical use:
        ctx = RunningContext(max_tokens=4000)
        ctx.add("intro", "We defined: ...")
        ctx.add("concept_a", "Concept A means ...")

        # in the next sub-task:
        prior = ctx.get_context()
        prompt = f"Previous content: {prior}\\n\\nNow explain ..."
    """

    def __init__(self, max_tokens: int = 4000,
                 separator: str = "\n\n---\n\n"):
        self.max_tokens = max_tokens
        self.separator = separator
        self._sections: list[_Section] = []

    def add(self, section_id: str, summary: str,
            full_text: str = "") -> None:
        """Add a section summary.

        Args:
            section_id: unique ID of the section
            summary: short summary (typically 100-300 words)
            full_text: optionally the full text (for later re-summarisation)
        """
        # If the section ID already exists: replace it
        for i, s in enumerate(self._sections):
            if s.id == section_id:
                self._sections[i] = _Section(
                    id=section_id, summary=summary, full_text=full_text,
                )
                return
        self._sections.append(_Section(
            id=section_id, summary=summary, full_text=full_text,
        ))

    def get_context(self, up_to_section: Optional[str] = None,
                    exclude: Optional[list[str]] = None) -> str:
        """Return the accumulated context.

        Args:
            up_to_section: only include sections up to this ID
            exclude: list of section IDs to leave out

        Returns:
            The concatenated context, shortened if needed to stay within max_tokens
        """
        sections = self._sections[:]

        if up_to_section:
            cut = None
            for i, s in enumerate(sections):
                if s.id == up_to_section:
                    cut = i
                    break
            if cut is not None:
                sections = sections[:cut]

        if exclude:
            sections = [s for s in sections if s.id not in exclude]

        if not sections:
            return ""

        # Keep within the token budget
        max_chars = self.max_tokens * CHARS_PER_TOKEN
        formatted = self._format(sections)

        if len(formatted) <= max_chars:
            return formatted

        # Too long: shorten older sections aggressively
        return self._truncate_to_budget(sections, max_chars)

    def _format(self, sections: list[_Section]) -> str:
        """Format a list of sections."""
        parts = []
        for s in sections:
            parts.append(f"[{s.id}]\n{s.summary}")
        return self.separator.join(parts)

    def _truncate_to_budget(self, sections: list[_Section],
                            max_chars: int) -> str:
        """Aggressively shorten older sections, starting from the front.

        Strategy: the newer sections matter more for the immediate
        context. The oldest sections are compressed to one sentence. If
        that is not enough, the oldest are removed entirely — until the
        newest fit into the budget.
        """
        if not sections:
            return ""

        # Step 1: shorten older sections aggressively
        full_keep = max(1, len(sections) // 2)
        recent = sections[-full_keep:]
        older = sections[:-full_keep]

        compressed_older = []
        for s in older:
            first_sentence = s.summary.split(".")[0][:80]
            compressed_older.append(_Section(
                id=s.id, summary=first_sentence + "...",
            ))

        all_sections = compressed_older + recent
        result = self._format(all_sections)

        if len(result) <= max_chars:
            return result

        # Step 2: still too long → remove the oldest entirely
        # until the newest fit
        while len(all_sections) > 1:
            all_sections = all_sections[1:]
            result = self._format(all_sections)
            if len(result) <= max_chars:
                return "[... older sections shortened]\n\n" + result

        # Step 3: even the last section is too long → hard cut;
        # keep the END of the last section (usually the more important part)
        last_section = sections[-1]
        truncated = last_section.summary[-max_chars:]
        return f"[... shortened]\n[{last_section.id}]\n{truncated}"

    def clear(self) -> None:
        """Remove all sections."""
        self._sections.clear()

    @property
    def section_count(self) -> int:
        return len(self._sections)

    @property
    def section_ids(self) -> list[str]:
        return [s.id for s in self._sections]

    def estimated_tokens(self) -> int:
        """Estimated token count of the current context."""
        return sum(s.estimated_tokens for s in self._sections)

    def serialize(self) -> dict:
        """For persistence."""
        return {
            "max_tokens": self.max_tokens,
            "separator": self.separator,
            "sections": [
                {"id": s.id, "summary": s.summary, "full_text": s.full_text}
                for s in self._sections
            ],
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RunningContext":
        """Restore from the persisted form."""
        ctx = cls(
            max_tokens=data.get("max_tokens", 4000),
            separator=data.get("separator", "\n\n---\n\n"),
        )
        for s in data.get("sections", []):
            ctx._sections.append(_Section(
                id=s["id"],
                summary=s["summary"],
                full_text=s.get("full_text", ""),
            ))
        return ctx
