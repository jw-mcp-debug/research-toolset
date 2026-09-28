"""
Synthesis: map phase (map + reduce)
===================================

Prompt for the map phase of the synthesis: per research question, the
associated extracts are condensed into a compact, coherent answer. The
answer blocks produced this way then go into the reduce phase
(`SYNTHESIS_PROMPT`).

Why map + reduce: with many extracts (hundreds are common), a single
call with ALL extracts in one prompt overflows the model's context window
(`ContextWindowExceededError`). With map + reduce every phase has a
clearly bounded input that fits into the context.

Caller: `src/pipeline/orchestrator.py::ResearchOrchestrator._run_synthesis`
"""

QUESTION_ANSWER_PROMPT = """You are a research assistant. Condense the
following extracts into ONE connected, readable answer to
the research question. This answer later becomes part of a larger
report — it should be understandable on its own, but must have
NO title and NO introduction.

RESEARCH QUESTION:
{question_id}: {question_text}

EXTRACTS (sorted by reliability):
{extracts}

Guidelines:

1. Write in connected PARAGRAPHS, not in bullet lists.
   Combine related facts into an argument or a description.

2. **Source references as Markdown links directly in the text** — e.g.
   "According to the [NVIDIA docs](https://...) the price is...".
   EVERY factual claim needs a source. Several pieces of evidence for
   the same statement may be combined in one sentence
   ("[source A](URL), [source B](URL)").

3. If extracts contradict each other: present this transparently
   ("X states Y, whereas Z reports W…") — do not shorten
   anything that is contradictory.

4. If extracts are marked "[POSITIVE]", "[NEGATIVE]" or "[META]":
   respect the polarity. POSITIVE = confirming the question,
   NEGATIVE = negating the question, META = a note on the source situation itself.
   For NEGATIVE extracts, make clear that the question cannot easily
   be answered in the affirmative.

5. Reliability: prefer statements rated "high", but
   do not hide those rated "low" — mark them as uncertain
   ("possibly…", "a single source states…").

6. NO invention. If the extracts do not answer a question,
   say so honestly: "The sources researched give no
   information on this."

7. Length: 200–600 words. With very many extracts on the same
   question you may write somewhat more, but be dense — do not repeat
   the same point from several extracts.

Write ONLY the answer, no meta introduction like "Here is the answer:"."""


__all__ = ["QUESTION_ANSWER_PROMPT"]
