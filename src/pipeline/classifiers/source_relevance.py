"""
Source relevance classifier (batched).

Judges for each SearchResult, before fetching, whether the source is
likely to contribute. An LLM classifier is used instead of DOM/URL
patterns for "listing pages", because such patterns wrongly discard
legitimate overview pages on the topic asked about.

Per search round 10-50 SearchResults → batched into 5 items per call.
Unproblematic with local LLM infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from src.core.limits import CLASSIFIER_BATCH_SIZE, CLASSIFIER_MAX_PARALLEL
from src.llm.json_parser import coerce_float, parse_llm_json
from src.pipeline.classifiers.base import (
    ClassifierCall,
    ClassifierLLM,
    ClassifierResult,
    hash_prompt,
)

logger = logging.getLogger(__name__)


VALID_RELEVANCE = {"high", "medium", "low", "off_topic"}


@dataclass
class SourceRelevance(ClassifierResult):
    """Result of `judge_source_relevance` for a single SearchResult."""
    relevance: str = "medium"        # high | medium | low | off_topic
    index: int = 0                   # Original-Index im Batch

    def should_fetch(self) -> bool:
        """Do not fetch only if 'off_topic' AND confidence > 0.7."""
        if self.relevance == "off_topic" and self.confidence > 0.7:
            return False
        return True


SOURCE_RELEVANCE_PROMPT = """For each of the following search results, assess whether it will contribute
to answering the research questions.

RESEARCH QUESTIONS:
{questions_text}

SEARCH RESULTS ({n} items):
{results_with_index}

Per result:
- "high": very probably a primary, focused answer to one of the questions
  (e.g. official datasheet, white paper, focused article)
- "medium": plausibly relevant, but probably only partly
  (e.g. a general overview article that touches on the topic)
- "low": marginal — might help, but the effort probably exceeds the benefit
  (e.g. forum discussion, secondary blog with an unclear level of detail)
- "off_topic": clearly not relevant
  (e.g. a listing page with dozens of other topics, a wrong product with a similar name,
   a marketing overview without detailed content)

Note:
- Listing/collection pages (URL patterns like /tags/, /category/, /search?q=, /index)
  are often "off_topic" because they mix content on different topics.
  BUT: only if the snippet shows that it actually is mixed. An
  overview page ON the topic asked about can be "medium".
- The snippet is informative but shorter than the real page. With sparse
  information: rather "medium" than "low".

Answer as a JSON array:
[
  {{
    "index": 1,
    "relevance": "high" | "medium" | "low" | "off_topic",
    "confidence": 0.0 to 1.0,
    "reasoning": "short justification"
  }},
  ...
]
"""


def _format_search_results_for_prompt(results: list) -> str:
    """Format SearchResults for the prompt.

    Accepts SearchResult objects (with title/url/snippet) and dicts.
    """
    out = []
    for i, r in enumerate(results, 1):
        if isinstance(r, dict):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
        else:
            title = getattr(r, "title", "")
            url = getattr(r, "url", "")
            snippet = getattr(r, "snippet", "")
        out.append(
            f"[{i}] Title: {title}\n"
            f"     URL: {url}\n"
            f"     Snippet: {snippet[:500]}"
        )
    return "\n\n".join(out)


def _format_questions_for_prompt(questions: list) -> str:
    """Format ResearchQuestions for the prompt.

    Accepts dicts and ResearchQuestion objects.
    """
    out = []
    for q in questions:
        if isinstance(q, dict):
            qid = q.get("id", "?")
            text = q.get("question", "")
        else:
            qid = getattr(q, "id", "?")
            text = getattr(q, "question", "")
        out.append(f"  {qid}: {text}")
    return "\n".join(out) if out else "(none)"


async def _judge_one_batch(
    questions_text: str,
    results: list,
    llm: ClassifierLLM,
    batch_offset: int,
) -> tuple[list[SourceRelevance], ClassifierCall]:
    """Judge a batch of 1..N SearchResults."""
    results_text = _format_search_results_for_prompt(results)
    prompt = SOURCE_RELEVANCE_PROMPT.format(
        questions_text=questions_text,
        n=len(results),
        results_with_index=results_text,
    )

    # call_classifier expects dict answers. Here the answer is an array,
    # so we use parse_llm_json directly.
    import time
    started = time.monotonic()
    call = ClassifierCall(
        name="judge_source_relevance",
        prompt_hash=hash_prompt(prompt),
        input_summary={"batch_offset": batch_offset, "n": len(results)},
    )

    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
    except Exception as e:
        call.fallback_used = True
        call.fallback_reason = f"llm_call_failed: {e}"
        call.duration_seconds = time.monotonic() - started
        logger.warning("judge_source_relevance: LLM call failed: %s", e)
        # Fallback: all "medium" (fetch to be safe)
        return [
            SourceRelevance(
                relevance="medium",
                confidence=0.0,
                index=batch_offset + i,
                reasoning="LLM fallback — default 'medium' (will be fetched)",
                fallback_used=True,
            )
            for i in range(len(results))
        ], call

    call.raw_response = raw[:1000]
    parsed = parse_llm_json(raw, default=[])
    call.duration_seconds = time.monotonic() - started

    if not isinstance(parsed, list) or not parsed:
        call.fallback_used = True
        call.fallback_reason = "no_array_in_response"
        return [
            SourceRelevance(
                relevance="medium",
                confidence=0.0,
                index=batch_offset + i,
                reasoning="JSON parse error — default 'medium'",
                fallback_used=True,
            )
            for i in range(len(results))
        ], call

    # Per-item mapping. The LLM might return items in a different order or
    # with wrong indices; we map by position and ignore the LLM's index.
    # That is more robust than relying on the index value.
    out: list[SourceRelevance] = []
    for i in range(len(results)):
        item = parsed[i] if i < len(parsed) else {}
        if not isinstance(item, dict):
            item = {}
        rel = str(item.get("relevance", "medium")).lower().strip()
        if rel not in VALID_RELEVANCE:
            rel = "medium"
        out.append(SourceRelevance(
            relevance=rel,
            confidence=coerce_float(item.get("confidence", 0.0)),
            index=batch_offset + i,
            reasoning=str(item.get("reasoning", "") or "").strip(),
        ))

    call.output = {"n_judged": len(out)}
    return out, call


async def judge_source_relevance(
    questions: list,
    search_results: list,
    llm: ClassifierLLM,
    batch_size: int = CLASSIFIER_BATCH_SIZE,
    max_parallel: int = CLASSIFIER_MAX_PARALLEL,
) -> tuple[list[SourceRelevance], list[ClassifierCall]]:
    """Judge all SearchResults in batches against the research questions.

    Args:
        questions: list of ResearchQuestion or dicts with id/question.
        search_results: list of SearchResult or dicts with
            title/url/snippet.
        llm: LLM client.
        batch_size: items per call (5).
        max_parallel: maximum parallel batch calls.

    Returns:
        (judgments, calls) — list of SourceRelevance in input order +
        list of classifier call logs (one per batch).
    """
    if not search_results:
        return [], []

    questions_text = _format_questions_for_prompt(questions)

    # Split into batches
    batches = []
    for i in range(0, len(search_results), batch_size):
        batches.append((i, search_results[i:i + batch_size]))

    sem = asyncio.Semaphore(max_parallel)

    async def run_batch(offset: int, items: list):
        async with sem:
            return await _judge_one_batch(questions_text, items, llm, offset)

    batch_results = await asyncio.gather(*[
        run_batch(offset, items) for offset, items in batches
    ])

    all_judgments: list[SourceRelevance] = []
    all_calls: list[ClassifierCall] = []
    for judgments, call in batch_results:
        all_judgments.extend(judgments)
        all_calls.append(call)

    # Make sure we return exactly as many as came in
    while len(all_judgments) < len(search_results):
        # Should never happen — if it does: default "medium"
        all_judgments.append(SourceRelevance(
            relevance="medium",
            confidence=0.0,
            index=len(all_judgments),
            reasoning="batch mapping error — default 'medium'",
            fallback_used=True,
        ))

    return all_judgments[:len(search_results)], all_calls
