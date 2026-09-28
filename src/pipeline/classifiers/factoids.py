"""
Factoid classifiers.

Candidates for hallucinations (DOIs, version numbers, amounts of money,
...) are not found with regex patterns but by factoid extraction plus
verification against the extracts.

Two-stage process:
    1. extract_factoids — identifies concrete individual claims in the report
    2. verify_factoid_against_extracts — checks whether each factoid is
       anchored in the extracts (batched, 5 per call)

Factoids that are not anchored are marked in the report:
    "verified=false" → claim not anchored in the extracts
    "verified=uncertain" → evidence unclear
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from src.core.limits import CLASSIFIER_BATCH_SIZE
from src.llm.json_parser import parse_llm_json
from src.pipeline.classifiers.base import (
    ClassifierCall,
    ClassifierLLM,
    ClassifierResult,
    hash_prompt,
)

logger = logging.getLogger(__name__)


# ─── Data models ──────────────────────────────────────────────────


VALID_FACTOID_TYPES = {
    "numeric_spec", "date", "identifier", "name", "claim", "quote",
}

VALID_VERIFIED = {"true", "false", "uncertain"}


@dataclass
class Factoid:
    """A single factoid from a report."""
    factoid: str
    type: str = "claim"           # numeric_spec | date | identifier | name | claim | quote
    report_position: str = ""

    def to_dict(self) -> dict:
        return {
            "factoid": self.factoid,
            "type": self.type,
            "report_position": self.report_position,
        }


@dataclass
class FactoidVerification(ClassifierResult):
    """Result of `verify_factoid_against_extracts` for one factoid."""
    factoid_index: int = 0
    verified: str = "uncertain"           # "true" | "false" | "uncertain"
    supporting_extract_id: str = ""
    supporting_quote: str = ""

    def is_verified(self) -> bool:
        return self.verified == "true"

    def is_unverified(self) -> bool:
        return self.verified == "false" and self.confidence > 0.7

    @classmethod
    def from_dict(cls, d: dict) -> "FactoidVerification":
        """Read a FactoidVerification from a dict (LLM JSON).

        Only does the plain conversion. Domain validation (verified ∈
        {true, false, uncertain}, confidence threshold that forces
        "uncertain") stays with the caller, because it is a rule of the
        verification step and not something the conversion layer can
        decide.

        `factoid_index` is set by the caller (index within the batch),
        because that information comes from the surrounding loop.
        """
        if not isinstance(d, dict):
            return cls()
        from src.pipeline.classifiers.base import coerce_float
        return cls(
            verified=str(d.get("verified", "uncertain") or "uncertain").lower().strip(),
            confidence=coerce_float(d.get("confidence", 0.0)),
            supporting_extract_id=str(
                d.get("supporting_extract_id", "") or "").strip(),
            supporting_quote=str(
                d.get("supporting_quote", "") or "").strip(),
            reasoning=str(d.get("reasoning", "") or "").strip(),
        )


# ─── Prompts ───────────────────────────────────────────────────────


FACTOID_EXTRACT_PROMPT = """Identify all FACTOIDS in the following report. Factoids are concrete,
checkable individual statements — typically with figures, names, dates,
version numbers, amounts, references, DOIs, proper names.

NOT factoids are:
- general statements ("AI is important", "many researchers work on it")
- judgements without a claim to evidence ("that is efficient")
- explanations, definitions, methodological sentences

REPORT:
{report}

Answer as a JSON list (typically 20-50 entries):
[
  {{
    "factoid": "The DGX B300 typically draws 14 kW",
    "type": "numeric_spec",
    "report_position": "section 'Power consumption', first paragraph"
  }},
  {{
    "factoid": "Published on 17 October 2024",
    "type": "date",
    "report_position": "..."
  }},
  {{
    "factoid": "DOI 10.1145/3404835.3463261",
    "type": "identifier",
    "report_position": "..."
  }}
]

Quote each factoid in the wording and language of the report.

Types:
- "numeric_spec": a concrete number with a unit
- "date": date / period
- "identifier": DOI, ISBN, GitHub repo, ID
- "name": proper name (person, organisation, product)
- "claim": factual claim without a number ("X led to the introduction of Y")
- "quote": verbatim quotation
"""


FACTOID_VERIFY_PROMPT = """For each of the following factoids, check whether it is anchored in the extracts
(i.e. does it have a reliable source in the researched extracts?).

FACTOIDS:
{factoids_with_index}

EXTRACTS (all, sorted by source):
{all_extracts}

Answer as a JSON array:
[
  {{
    "factoid_index": 1,
    "verified": "true" | "false" | "uncertain",
    "confidence": 0.0 to 1.0,
    "supporting_extract_id": "E47" or empty,
    "supporting_quote": "verbatim excerpt from the supporting extract" or empty,
    "reasoning": "..."
  }},
  ...
]

Rules:
- "verified=true": the factoid is clearly supported by at least one extract
  — exact number, exact date, exact name.
- "verified=false": the factoid can NOT be found in the extracts, or
  the extracts give a different number/date/name. That is a clear
  suspicion of hallucination.
- "verified=uncertain": the extracts mention the topic, but not the
  concrete detail. A suspicion of hallucination is possible but cannot be proven.

With a confidence < 0.5: prefer "uncertain".
"""


# ─── extract_factoids ──────────────────────────────────────────────


async def extract_factoids(
    report: str,
    llm: ClassifierLLM,
    max_tokens: int = 4096,
) -> tuple[list[Factoid], ClassifierCall]:
    """Identify factoids in the synthesis report.

    Returns:
        (factoids, call) — an empty list on fallback, with
        call.fallback_used=True.
    """
    started = time.monotonic()
    prompt = FACTOID_EXTRACT_PROMPT.format(report=report[:30_000])
    call = ClassifierCall(
        name="extract_factoids",
        prompt_hash=hash_prompt(prompt),
        input_summary={"report_length": len(report)},
    )

    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}], max_tokens=max_tokens,
        )
    except Exception as e:
        call.fallback_used = True
        call.fallback_reason = f"llm_call_failed: {e}"
        call.duration_seconds = time.monotonic() - started
        return [], call

    call.raw_response = raw[:1000]
    parsed = parse_llm_json(raw, default=[])
    call.duration_seconds = time.monotonic() - started

    if not isinstance(parsed, list):
        call.fallback_used = True
        call.fallback_reason = "no_array_in_response"
        return [], call

    factoids: list[Factoid] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = str(item.get("factoid", "") or "").strip()
        if not text:
            continue
        ftype = str(item.get("type", "claim")).lower().strip()
        if ftype not in VALID_FACTOID_TYPES:
            ftype = "claim"
        factoids.append(Factoid(
            factoid=text,
            type=ftype,
            report_position=str(item.get("report_position", "") or "").strip(),
        ))

    call.output = {"n_factoids": len(factoids)}
    return factoids, call


# ─── verify_factoid_against_extracts ───────────────────────────────


def _format_factoids_for_prompt(factoids: list[Factoid], offset: int) -> str:
    out = []
    for i, f in enumerate(factoids, 1):
        out.append(f"[{offset + i}] ({f.type}) {f.factoid}")
    return "\n".join(out)


def _format_extracts_for_verify(extracts: list) -> str:
    """Format a list of SourceExtracts for the verify prompt."""
    out = []
    for i, e in enumerate(extracts, 1):
        if isinstance(e, dict):
            ext_id = e.get("id") or f"E{i}"
            fact = e.get("fact", "")
            source = e.get("source_title") or e.get("source_url", "?")
        else:
            ext_id = getattr(e, "id", None) or f"E{i}"
            fact = getattr(e, "fact", "")
            source = (getattr(e, "source_title", "")
                      or getattr(e, "source_url", "?"))
        out.append(f"[{ext_id}] {fact} (source: {source})")
    return "\n".join(out) if out else "(none)"


async def _verify_one_batch(
    factoids: list[Factoid],
    extracts_text: str,
    llm: ClassifierLLM,
    batch_offset: int,
) -> tuple[list[FactoidVerification], ClassifierCall]:
    started = time.monotonic()
    factoids_text = _format_factoids_for_prompt(factoids, batch_offset)
    prompt = FACTOID_VERIFY_PROMPT.format(
        factoids_with_index=factoids_text,
        all_extracts=extracts_text,
    )
    call = ClassifierCall(
        name="verify_factoid_against_extracts",
        prompt_hash=hash_prompt(prompt),
        input_summary={"batch_offset": batch_offset, "n": len(factoids)},
    )

    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}], max_tokens=2048,
        )
    except Exception as e:
        call.fallback_used = True
        call.fallback_reason = f"llm_call_failed: {e}"
        call.duration_seconds = time.monotonic() - started
        return [
            FactoidVerification(
                factoid_index=batch_offset + i + 1,
                verified="uncertain",
                confidence=0.0,
                reasoning="LLM-Fallback",
                fallback_used=True,
            )
            for i in range(len(factoids))
        ], call

    call.raw_response = raw[:1000]
    parsed = parse_llm_json(raw, default=[])
    call.duration_seconds = time.monotonic() - started

    if not isinstance(parsed, list):
        call.fallback_used = True
        call.fallback_reason = "no_array_in_response"
        return [
            FactoidVerification(
                factoid_index=batch_offset + i + 1,
                verified="uncertain",
                confidence=0.0,
                reasoning="JSON parse error",
                fallback_used=True,
            )
            for i in range(len(factoids))
        ], call

    out: list[FactoidVerification] = []
    for i in range(len(factoids)):
        item = parsed[i] if i < len(parsed) else {}
        if not isinstance(item, dict):
            item = {}
        # Conversion from JSON comes from from_dict; validation stays here
        # (allowed verified values + forcing by confidence).
        fv = FactoidVerification.from_dict(item)
        fv.factoid_index = batch_offset + i + 1
        if fv.verified not in VALID_VERIFIED:
            fv.verified = "uncertain"
        if fv.confidence < 0.5 and fv.verified != "uncertain":
            # Rule: confidence < 0.5 → "uncertain"
            fv.verified = "uncertain"
        out.append(fv)

    call.output = {"n_verified": len(out)}
    return out, call


async def verify_factoids_against_extracts(
    factoids: list[Factoid],
    extracts: list,
    llm: ClassifierLLM,
    batch_size: int = CLASSIFIER_BATCH_SIZE,
    max_parallel: int = 4,
) -> tuple[list[FactoidVerification], list[ClassifierCall]]:
    """Verify factoids against the collected extracts.

    Batched (5 factoids per call) and parallelised.
    """
    if not factoids:
        return [], []

    extracts_text = _format_extracts_for_verify(extracts)

    batches = []
    for i in range(0, len(factoids), batch_size):
        batches.append((i, factoids[i:i + batch_size]))

    sem = asyncio.Semaphore(max_parallel)

    async def run_batch(offset: int, items: list[Factoid]):
        async with sem:
            return await _verify_one_batch(items, extracts_text, llm, offset)

    batch_results = await asyncio.gather(*[
        run_batch(offset, items) for offset, items in batches
    ])

    all_verifications: list[FactoidVerification] = []
    all_calls: list[ClassifierCall] = []
    for verifs, call in batch_results:
        all_verifications.extend(verifs)
        all_calls.append(call)

    return all_verifications, all_calls
