"""
Pipeline nodes for the classifier layer.

These nodes wrap the orchestrator's classifier calls (`_classify_*`
methods, `_filter_off_topic_sources`, `_verify_report_factoids`) as
testable pipeline nodes.

Order in the pipeline:
  1. QueryAnchorNode          — before the harvest, once per run
  2. OffTopicFilterNode       — after search+fetch, before the harvest
  3. CoverageNode             — after the harvest, per round
  4. ContinueDecisionNode     — after coverage, per round
  5. DiagnosisNode            — before the synthesis, once
  6. FactoidVerificationNode  — after the synthesis, once

Nodes read from / write to the `HarvestContext` and use the tested
classifier functions from `src.pipeline.classifiers`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.output_language import prompt_language_line
from src.output_language import t as catalog_t
from src.pipeline.dag import (
    FailMode,
    NodeResult,
    NodeStatus,
    PipelineNode,
)

if TYPE_CHECKING:
    from src.llm.client import DualLLMClient

logger = logging.getLogger(__name__)


# When is a "report" a report at all?
# Header + request + metadata footer add up to about 260 characters
# without any content — an artefact that could otherwise be exported
# as a finished report. Anything below this threshold counts as
# degenerate; it is not revised any further but reported.
_MIN_REPORT_CHARS = 600


# ── 1. QueryAnchorNode ────────────────────────────────────────────


class QueryAnchorNode(PipelineNode):
    """Classify the research query for a person/organisation anchor.

    Mutations of ctx:
      - ctx.query_anchor (set by the node)
      - ctx.classifier_calls (appended)

    Use-case override: if the use case does not have `query_anchor` in
    `classifiers_enabled` (e.g. product_comparison), the node is skipped.
    """

    name = "query_anchor"
    fail_mode = FailMode.CONTINUE  # on failure: no filter, the pipeline runs

    def __init__(self, llm: "DualLLMClient"):
        self.llm = llm

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        # If already classified (e.g. by the format agent beforehand): not
        # again — the classifier is deterministic enough, a re-run would be
        # wasteful.
        if getattr(ctx, "query_anchor", None) is not None:
            return False, "query_anchor already set"
        return True, ""

    async def run(self, ctx: Any) -> dict:
        from src.llm.classifier_adapter import HarvestModelAdapter
        from src.pipeline.classifiers.query_anchor import (
            classify_query_anchor,
        )

        plan_summary = ""
        if getattr(ctx, "research_plan", None) is not None:
            plan_summary = ctx.research_plan.summary or ""

        adapter = HarvestModelAdapter(self.llm)
        anchor, call = await classify_query_anchor(
            query=ctx.query,
            plan_summary=plan_summary,
            llm=adapter,
        )

        # ctx mutation
        ctx.query_anchor = anchor
        if hasattr(ctx, "classifier_calls"):
            ctx.classifier_calls.append(call)
        else:
            ctx.classifier_calls = [call]

        return {
            "anchor_type": anchor.type,
            "target": anchor.target,
            "confidence": anchor.confidence,
            "fallback_used": anchor.fallback_used,
        }


# ── 2. OffTopicFilterNode ─────────────────────────────────────────


class OffTopicFilterNode(PipelineNode):
    """Filter thematically unsuitable sources via judge_source_relevance.

    Mutations of ctx:
      - ctx.sources (filtered)
      - ctx.classifier_calls (appended)

    Contract: only sources with `relevance='off_topic'` AND
    `confidence>0.7` are excluded — everything else stays.
    """

    name = "off_topic_filter"
    fail_mode = FailMode.CONTINUE  # on failure: keep all sources

    def __init__(self, llm: "DualLLMClient"):
        self.llm = llm

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        if not getattr(ctx, "sources", None):
            return False, "no sources"
        plan = getattr(ctx, "research_plan", None)
        if plan is None or not plan.questions:
            return False, "no research plan"
        return True, ""

    async def run(self, ctx: Any) -> dict:
        from src.llm.classifier_adapter import HarvestModelAdapter
        from src.pipeline.classifiers.source_relevance import (
            judge_source_relevance,
        )

        sources = ctx.sources
        plan = ctx.research_plan

        questions_payload = [
            {"id": q.id, "question": q.question} for q in plan.questions
        ]
        results_payload = [
            {
                "title": s.title or "",
                "url": s.url,
                "snippet": (s.content or "")[:500],
            }
            for s in sources
        ]

        adapter = HarvestModelAdapter(self.llm)
        judgments, calls = await judge_source_relevance(
            questions=questions_payload,
            search_results=results_payload,
            llm=adapter,
        )

        # Log the calls
        if hasattr(ctx, "classifier_calls"):
            ctx.classifier_calls.extend(calls)
        else:
            ctx.classifier_calls = list(calls)

        # Apply the filter — no zip(): a missing verdict means keep
        # (fail-open, consistent with this node's fail_mode=CONTINUE).
        if len(judgments) != len(sources):
            logger.warning(
                "OffTopicFilter: %d verdicts for %d sources — "
                "missing ones are kept", len(judgments), len(sources),
            )
        kept = []
        rejected_count = 0
        for i, src in enumerate(sources):
            j = judgments[i] if i < len(judgments) else None
            if j is None or j.should_fetch():
                kept.append(src)
            else:
                rejected_count += 1
                logger.info(
                    "OffTopicFilter: %s (rel=%s, conf=%.2f)",
                    src.url, j.relevance, j.confidence,
                )

        ctx.sources = kept
        return {
            "rejected": rejected_count,
            "kept": len(kept),
            "total": len(sources),
        }


# ── 3. CoverageNode ───────────────────────────────────────────────


class CoverageNode(PipelineNode):
    """Assess coverage per question via evaluate_coverage.

    Mutations of ctx:
      - ctx.coverage_per_round (appended)
      - ctx.classifier_calls (appended)
    """

    name = "coverage"
    fail_mode = FailMode.CONTINUE

    def __init__(
        self, llm: "DualLLMClient",
        round_number: int,
    ):
        self.llm = llm
        self.round_number = round_number

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        plan = getattr(ctx, "research_plan", None)
        if plan is None or not plan.questions:
            return False, "no research plan"
        return True, ""

    async def run(self, ctx: Any) -> dict:
        import asyncio
        from src.llm.classifier_adapter import HarvestModelAdapter
        from src.pipeline.classifiers.coverage import evaluate_coverage

        # Read this round's active filters from filter_stats_per_round
        active_filters: list[str] = []
        max_loss_rate = 0.0
        filter_stats_list = getattr(ctx, "filter_stats_per_round", None) or []
        if filter_stats_list:
            current = filter_stats_list[-1]
            for fname, fstats in current.items():
                if fstats.get("activated"):
                    active_filters.append(fname)
                    max_loss_rate = max(
                        max_loss_rate, fstats.get("loss_rate", 0.0),
                    )

        adapter = HarvestModelAdapter(self.llm)

        async def _eval_one(q):
            extracts_for_q = [
                {
                    "id": f"E{i}",
                    "fact": e.fact,
                    "source_title": e.source_title or "",
                    "reliability": e.reliability,
                }
                for i, e in enumerate(ctx.extracts)
                if e.question_id == q.id
                and getattr(e, "polarity", "positive") == "positive"
            ]
            return await evaluate_coverage(
                question_id=q.id,
                question=q.question,
                priority=q.priority,
                extracts=extracts_for_q,
                active_filters=active_filters,
                filter_loss_rate=max_loss_rate,
                llm=adapter,
            )

        questions = ctx.research_plan.questions
        coverage_tasks = [_eval_one(q) for q in questions]
        outcomes = await asyncio.gather(*coverage_tasks, return_exceptions=True)

        coverage_map: dict[str, dict] = {}
        for q, outcome in zip(questions, outcomes):
            if isinstance(outcome, BaseException):
                logger.warning("CoverageNode: %s — %s", q.id, outcome)
                continue
            result, call = outcome
            if hasattr(ctx, "classifier_calls"):
                ctx.classifier_calls.append(call)
            coverage_map[q.id] = {
                "coverage": result.coverage,
                "confidence": result.confidence,
                "missing_aspects": result.missing_aspects,
                "supporting_extract_ids": result.supporting_extract_ids,
                "reasoning": result.reasoning,
                "fallback_used": result.fallback_used,
            }

        if not hasattr(ctx, "coverage_per_round"):
            ctx.coverage_per_round = []
        ctx.coverage_per_round.append({
            "round": self.round_number, "results": coverage_map,
        })
        return {
            "round": self.round_number,
            "n_evaluated": len(coverage_map),
        }


# ── 4. ContinueDecisionNode ───────────────────────────────────────


class ContinueDecisionNode(PipelineNode):
    """Decide via decide_continue_research whether another round is worthwhile.

    Reads the latest coverage assessment from ctx and returns a continue
    decision. With `decision != "continue"`, `STOP_PIPELINE` is returned —
    the engine ends the loop.

    Mutations of ctx:
      - ctx.continue_decisions (appended)
      - ctx.classifier_calls (appended)
    """

    name = "continue_decision"
    fail_mode = FailMode.CONTINUE

    def __init__(
        self,
        llm: "DualLLMClient",
        round_number: int,
        max_rounds: int,
        new_extracts: int,
        new_sources: int,
    ):
        self.llm = llm
        self.round_number = round_number
        self.max_rounds = max_rounds
        self.new_extracts = new_extracts
        self.new_sources = new_sources

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        if not getattr(ctx, "coverage_per_round", None):
            return False, "no coverage result"
        return True, ""

    async def run(self, ctx: Any):
        from src.llm.classifier_adapter import HarvestModelAdapter
        from src.pipeline.classifiers.coverage import (
            decide_continue_research,
        )

        # Prepare the coverage of the last round
        last_round = ctx.coverage_per_round[-1].get("results", {})
        counts = {"answered": 0, "partial": 0,
                  "unanswered": 0, "filter_blocked": 0}
        details_lines = []
        for qid, c in last_round.items():
            cov = c["coverage"]
            if cov in counts:
                counts[cov] += 1
            details_lines.append(
                f"  {qid}: {cov} (conf={c['confidence']:.2f})"
            )

        max_loss_rate = 0.0
        filter_stats_list = getattr(ctx, "filter_stats_per_round", None) or []
        if filter_stats_list:
            for fstats in filter_stats_list[-1].values():
                max_loss_rate = max(
                    max_loss_rate, fstats.get("loss_rate", 0.0),
                )

        adapter = HarvestModelAdapter(self.llm)
        result, call = await decide_continue_research(
            round_number=self.round_number,
            max_rounds=self.max_rounds,
            answered_count=counts["answered"],
            partial_count=counts["partial"],
            unanswered_count=counts["unanswered"],
            filter_blocked_count=counts["filter_blocked"],
            total_count=len(last_round),
            new_extracts=self.new_extracts,
            new_sources=self.new_sources,
            filter_loss_rate=max_loss_rate,
            total_extracts=len(getattr(ctx, "extracts", [])),
            total_sources=len(getattr(ctx, "sources", [])),
            coverage_details="\n".join(details_lines),
            llm=adapter,
        )

        if hasattr(ctx, "classifier_calls"):
            ctx.classifier_calls.append(call)

        if not hasattr(ctx, "continue_decisions"):
            ctx.continue_decisions = []
        ctx.continue_decisions.append({
            "round": self.round_number,
            "decision": result.decision,
            "confidence": result.confidence,
            "next_round_focus": result.next_round_focus,
            "reasoning": result.reasoning,
            "fallback_used": result.fallback_used,
        })

        # If the classifier says "stop": the engine should end the pipeline
        if result.decision != "continue":
            return NodeResult(
                status=NodeStatus.STOP_PIPELINE,
                node_name=self.name,
                metadata={
                    "decision": result.decision,
                    "reasoning": result.reasoning,
                },
            )
        return {
            "decision": result.decision,
            "confidence": result.confidence,
        }


# ── 5. DiagnosisNode ──────────────────────────────────────────────


class DiagnosisNode(PipelineNode):
    """Diagnose the final pipeline state via diagnose_pipeline_state.

    Mutations of ctx:
      - ctx.final_diagnosis (set)
      - ctx.classifier_calls (appended)
    """

    name = "diagnosis"
    fail_mode = FailMode.CONTINUE

    def __init__(
        self,
        llm: "DualLLMClient",
        max_rounds: int,
        use_case: str = "web",
    ):
        """
        Args:
            llm: LLM client.
            max_rounds: maximum number of research rounds (for the
                "rounds / max_rounds" information in the diagnosis prompt).
            use_case: the current mode (e.g. "web", "institution"). Passed
                ON to the diagnosis classifier, which uses it in the prompt
                to put its assessment in context (an institution search
                with 2 sources is to be judged differently from a web
                search with 2 sources).
                **No** profile lookup — pure context information.
        """
        self.llm = llm
        self.max_rounds = max_rounds
        self.use_case = use_case

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        return True, ""

    async def run(self, ctx: Any) -> dict:
        from src.llm.classifier_adapter import HarvestModelAdapter
        from src.pipeline.classifiers.diagnose import (
            diagnose_pipeline_state,
        )
        from src.pipeline.harvest_parser import count_by_polarity

        polarity_counts = count_by_polarity(getattr(ctx, "extracts", []))

        coverage_distribution = {
            "answered": 0, "partial": 0,
            "unanswered": 0, "filter_blocked": 0,
        }
        cov_per_round = getattr(ctx, "coverage_per_round", None) or []
        if cov_per_round:
            last = cov_per_round[-1].get("results", {})
            for c in last.values():
                cov = c.get("coverage", "partial")
                if cov in coverage_distribution:
                    coverage_distribution[cov] += 1

        filter_stats: dict = {}
        filter_stats_list = getattr(ctx, "filter_stats_per_round", None) or []
        if filter_stats_list:
            filter_stats = dict(filter_stats_list[-1])

        stop_reason = "stop_done"
        decisions = getattr(ctx, "continue_decisions", None) or []
        if decisions:
            last_decision = decisions[-1]["decision"]
            if last_decision != "continue":
                stop_reason = last_decision

        active_filters = [
            n for n, s in filter_stats.items() if s.get("activated")
        ]

        anchor = getattr(ctx, "query_anchor", None)
        anchor_type = anchor.type if anchor else "none"
        anchor_confidence = anchor.confidence if anchor else 0.0

        adapter = HarvestModelAdapter(self.llm)
        result, call = await diagnose_pipeline_state(
            query=ctx.query,
            use_case=self.use_case,
            n_rounds=getattr(ctx, "rounds_completed", 0),
            max_rounds=self.max_rounds,
            stop_reason=stop_reason,
            final_extracts=len(getattr(ctx, "extracts", [])),
            positive_extracts=polarity_counts["positive"],
            negative_extracts=polarity_counts["negative"],
            meta_extracts=polarity_counts["meta"],
            n_sources=len(getattr(ctx, "sources", [])),
            coverage_distribution=coverage_distribution,
            filter_stats=filter_stats,
            anchor_type=anchor_type,
            anchor_confidence=anchor_confidence,
            active_filters=active_filters,
            llm=adapter,
        )

        if hasattr(ctx, "classifier_calls"):
            ctx.classifier_calls.append(call)

        ctx.final_diagnosis = {
            "diagnosis": result.diagnosis,
            "confidence": result.confidence,
            "remediation": result.remediation,
            "user_message": result.user_message,
            "reasoning": result.reasoning,
            "fallback_used": result.fallback_used,
            "is_problematic": result.is_problematic(),
            "is_successful": result.is_successful(),
        }
        logger.info(
            "🩺 Diagnosis node: %s (conf=%.2f)",
            result.diagnosis, result.confidence,
        )
        return {"diagnosis": result.diagnosis}


# ── 6. FactoidVerificationNode ────────────────────────────────────


class FactoidVerificationNode(PipelineNode):
    """Extract and verify factoids from the report.

    Mutations of ctx:
      - ctx.final_report (report possibly annotated with a consistency block)
      - ctx.factoid_verifications (set)
      - ctx.classifier_calls (appended)
    """

    name = "factoid_verification"
    fail_mode = FailMode.CONTINUE

    def __init__(self, llm: "DualLLMClient"):
        self.llm = llm

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        if not getattr(ctx, "final_report", "").strip():
            return False, "no report"
        return True, ""

    async def run(self, ctx: Any) -> dict:
        from src.llm.classifier_adapter import HarvestModelAdapter
        from src.pipeline.classifiers.factoids import (
            extract_factoids,
            verify_factoids_against_extracts,
        )
        from src.pipeline.orchestrator import _annotate_unverified_factoids

        adapter = HarvestModelAdapter(self.llm)

        # Extract
        try:
            factoids, ec = await extract_factoids(ctx.final_report, llm=adapter)
            if hasattr(ctx, "classifier_calls"):
                ctx.classifier_calls.append(ec)
        except Exception as e:
            logger.warning("FactoidExtraction failed: %s", e)
            return {"extracted": 0}

        if not factoids:
            ctx.factoid_verifications = []
            return {"extracted": 0}

        # Positive extracts as support
        positive_extracts = [
            {"id": f"E{i}", "fact": e.fact,
             "source_title": e.source_title or "",
             "source_url": e.source_url or ""}
            for i, e in enumerate(getattr(ctx, "extracts", []))
            if getattr(e, "polarity", "positive") == "positive"
        ]

        if not positive_extracts:
            ctx.factoid_verifications = [
                {"factoid": f.factoid, "type": f.type, "verified": "uncertain",
                 "supporting_extract_id": None, "supporting_quote": "",
                 "confidence": 0.0, "fallback_used": True}
                for f in factoids
            ]
            return {"extracted": len(factoids), "verified": 0}

        # Verify
        try:
            results, vcalls = await verify_factoids_against_extracts(
                factoids, positive_extracts, llm=adapter,
            )
            if hasattr(ctx, "classifier_calls"):
                ctx.classifier_calls.extend(vcalls)
        except Exception as e:
            logger.warning("FactoidVerification failed: %s", e)
            return {"extracted": len(factoids), "verified": 0}

        verifications = []
        unverified_count = 0
        for f, r in zip(factoids, results):
            verifications.append({
                "factoid": f.factoid, "type": f.type,
                "verified": r.verified,
                "supporting_extract_id": r.supporting_extract_id,
                "supporting_quote": r.supporting_quote,
                "confidence": r.confidence,
                "fallback_used": r.fallback_used,
            })
            if r.is_unverified():
                unverified_count += 1

        ctx.final_report = _annotate_unverified_factoids(
            ctx.final_report, verifications,
            lang=getattr(ctx, "output_language", None),
        )
        ctx.factoid_verifications = verifications
        return {
            "extracted": len(factoids),
            "verified": len(factoids) - unverified_count,
            "unverified": unverified_count,
        }


# ── 6b. ReportRevisionNode ────────────────────────────────────────


class ReportRevisionNode(PipelineNode):
    """Correct the report after the factoid verification.

    If the factoid verification classified statements with high
    confidence (≥0.9) as not supported by the sources, this node calls
    the LLM and has it correct the report **with an understanding of
    the meaning** — instead of marking things up with substring
    heuristics.

    With confidence < 0.9 this node does NOT intervene. Such cases are
    covered by the discreet notice block at the end of the report, which
    `_annotate_unverified_factoids` sets (see FactoidVerificationNode).

    Mutations of ctx:
      - ctx.final_report (replaced by the corrected version if a revision
        was carried out)
      - ctx.report_revision (set: what was done — transparency for the
        pipeline-run tab)

    Lifecycle:
      - runs AFTER FactoidVerificationNode
      - runs BEFORE ReportQualityNode/QueryFulfillmentNode (so that
        their assessment judges the corrected report)
      - applies_to skips itself if there are no high-confidence
        contradictions → no additional LLM call in the normal case
    """

    name = "report_revision"
    fail_mode = FailMode.CONTINUE
    HIGH_CONFIDENCE_THRESHOLD = 0.9

    def __init__(self, llm: "DualLLMClient"):
        self.llm = llm

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        report = getattr(ctx, "final_report", "") or ""
        if not report.strip():
            return False, "no report"
        if len(report.strip()) < _MIN_REPORT_CHARS:
            # A report that consists only of header and footer has nothing to
            # correct — the verifier would only cut the metadata down further.
            return False, "report too short for a revision"
        verifs = getattr(ctx, "factoid_verifications", []) or []
        high_conf_unverified = [
            v for v in verifs
            if v.get("verified") == "false"
            and v.get("confidence", 0) >= self.HIGH_CONFIDENCE_THRESHOLD
        ]
        if not high_conf_unverified:
            return False, ("no contradicted statements with "
                           "confidence >= 0.9 -> no revision needed")
        return True, ""

    async def run(self, ctx: Any) -> dict:
        import asyncio
        from src.prompts import REPORT_REVISION_PROMPT

        verifs = ctx.factoid_verifications
        high_conf = [
            v for v in verifs
            if v.get("verified") == "false"
            and v.get("confidence", 0) >= self.HIGH_CONFIDENCE_THRESHOLD
        ]

        # Assemble the correction block for the prompt
        unverified_lines = []
        for v in high_conf:
            unverified_lines.append(
                f"- STATEMENT IN THE REPORT: {v.get('factoid', '?')}\n"
                f"  TYPE: {v.get('type', '?')}\n"
                f"  VERIFIER CONFIDENCE: "
                f"{v.get('confidence', 0):.2f}"
            )
            quote = v.get("supporting_quote", "")
            if quote:
                unverified_lines.append(f"  SOURCE STATEMENT: {quote}")
            reasoning = v.get("reasoning", "")
            if reasoning:
                unverified_lines.append(f"  REASONING: {reasoning}")
            unverified_lines.append("")
        unverified_block = "\n".join(unverified_lines).strip()

        original_report = ctx.final_report
        prompt = REPORT_REVISION_PROMPT.format(
            report=original_report,
            unverified_block=unverified_block,
        )
        prompt += prompt_language_line(getattr(ctx, "output_language", None))

        # Run the revision — with a timeout, because a hanging LLM would
        # block the whole pipeline. 5 minutes is generous even for long
        # reports.
        try:
            corrected = await asyncio.wait_for(
                self.llm.primary_complete(
                    [{"role": "user", "content": prompt}],
                    max_tokens=16384,
                ),
                timeout=300,
            )
            corrected = (corrected or "").strip()
            if (not corrected
                    or len(corrected) < len(original_report) // 4
                    or len(corrected) < _MIN_REPORT_CHARS):
                # Suspiciously short → the LLM probably did not return the whole
                # report. Safety fallback.
                #
                # The relative bound alone is not enough: for an already degenerate
                # report (260 characters) it would be 65 characters, so a cut down
                # to 124 characters would pass unnoticed. Hence an absolute lower
                # bound as well.
                logger.warning(
                    "ReportRevision: corrected report suspiciously "
                    "short (%d characters vs. original %d) — keeping the "
                    "original",
                    len(corrected), len(original_report),
                )
                ctx.report_revision = {
                    "applied": False,
                    "reason": "corrected report suspiciously short",
                    "n_corrected": len(high_conf),
                    "factoids_corrected": [v.get("factoid", "")
                                           for v in high_conf],
                }
                return {"applied": False, "reason": "output_too_short"}

            ctx.final_report = corrected
            ctx.report_revision = {
                "applied": True,
                "n_corrected": len(high_conf),
                "factoids_corrected": [v.get("factoid", "")
                                       for v in high_conf],
                "original_length": len(original_report),
                "corrected_length": len(corrected),
            }
            logger.info(
                "ReportRevision: %d statements corrected "
                "(report: %d → %d characters)",
                len(high_conf), len(original_report), len(corrected),
            )
            return {
                "applied": True,
                "n_corrected": len(high_conf),
            }

        except asyncio.TimeoutError:
            logger.warning("ReportRevision: timeout — keeping the original")
            ctx.report_revision = {
                "applied": False,
                "reason": "Timeout (>300s)",
                "n_corrected": len(high_conf),
                "factoids_corrected": [v.get("factoid", "")
                                       for v in high_conf],
            }
            return {"applied": False, "reason": "timeout"}
        except Exception as e:
            logger.warning("ReportRevision: error %s — keeping the original",
                           type(e).__name__)
            ctx.report_revision = {
                "applied": False,
                "reason": f"LLM error: {type(e).__name__}",
                "n_corrected": len(high_conf),
                "factoids_corrected": [v.get("factoid", "")
                                       for v in high_conf],
            }
            return {"applied": False, "reason": "llm_error"}


# ═══════════════════════════════════════════════════════════════════
# Wrapper nodes for the structural pipeline phases
# ═══════════════════════════════════════════════════════════════════
#
# In contrast to the classifier nodes above, these nodes contain NO
# logic of their own — they call the orchestrator's `_run_*` methods
# and write the result into ctx. The established logic stays in one
# place; the nodes only give it a standardised shell.
#
# The lifecycle contracts (what is expected/produced in ctx) are
# documented in each node.
#
# The back-reference `self.orch` is explicit, so that it is clear
# these nodes delegate to the orchestrator.


# ── 7. FormatAgentNode ────────────────────────────────────────────


class FormatAgentNode(PipelineNode):
    """Wrapper for `ResearchOrchestrator._run_format_agent` (phase 0).

    Creates the output schema (which sections should the report have?).
    Without this schema the synthesis cannot run — fail_mode=STOP.

    Input (in ctx):
      - ctx.query
      - ctx.chat_history

    Output (in ctx):
      - ctx.output_schema

    Can be skipped if ctx.output_schema is already set (e.g. because a
    previous run already determined the schema — plan preview gate).
    """

    name = "format_agent"
    fail_mode = FailMode.STOP

    def __init__(
        self, orchestrator,
        context_docs: str = "",
        template_name: str = "",
    ):
        self.orch = orchestrator
        self.context_docs = context_docs
        self.template_name = template_name

    def applies_to(self, ctx):
        if getattr(ctx, "output_schema", None) is not None:
            return False, "output_schema already set"
        return True, ""

    async def run(self, ctx):
        result = await self.orch._run_format_agent(
            query=ctx.query,
            chat_history=ctx.chat_history or [],
            context_docs=self.context_docs,
            template_name=self.template_name,
        )
        ctx.output_schema = result
        return {
            "format": result.format_type if result else None,
            "sections": len(result.sections) if result else 0,
        }


# ── 8. AnalysisNode ───────────────────────────────────────────────


class AnalysisNode(PipelineNode):
    """Wrapper for `ResearchOrchestrator._run_analysis` (phase 1).

    Breaks the request down into a `ResearchPlan` (questions + search
    terms). No plan, no research — fail_mode=STOP.

    Input (in ctx):
      - ctx.query
      - ctx.output_schema (mandatory)

    Output (in ctx):
      - ctx.research_plan
    """

    name = "analysis"
    fail_mode = FailMode.STOP

    def __init__(self, orchestrator, context_docs: str = ""):
        self.orch = orchestrator
        self.context_docs = context_docs

    def applies_to(self, ctx):
        if getattr(ctx, "research_plan", None) is not None:
            return False, "research_plan already set"
        if getattr(ctx, "output_schema", None) is None:
            return False, "no output_schema (did FormatAgentNode run?)"
        return True, ""

    async def run(self, ctx):
        plan = await self.orch._run_analysis(
            query=ctx.query,
            context_docs=self.context_docs,
            output_schema=ctx.output_schema,
        )
        ctx.research_plan = plan

        # Plan validation. With blocking errors the node aborts the pipeline
        # via FailMode.STOP — otherwise the expensive research loop would run
        # on a broken plan and deliver a useless result. WARN issues are
        # logged but do not block.
        from src.pipeline.plan_validation import (
            format_issues_for_log,
            has_blocking_errors,
            validate_plan_conventions,
        )
        issues = validate_plan_conventions(plan)
        if issues:
            logger.info(
                "Plan validation:\n%s", format_issues_for_log(issues),
            )
        if has_blocking_errors(issues):
            error_msgs = [i.message for i in issues if i.is_error()]
            raise ValueError(
                "Plan does not conform: " + " | ".join(error_msgs)
            )

        return {
            "questions": len(plan.questions) if plan else 0,
            "direct_urls": len(plan.direct_urls) if plan else 0,
            "validation_warnings": sum(1 for i in issues if i.is_warning()),
        }


# ── 9. SearchAndFetchNode ─────────────────────────────────────────


class SearchAndFetchNode(PipelineNode):
    """Wrapper for `ResearchOrchestrator._run_search_and_fetch` (phase 2).

    Searches and fetches sources for one research round. The node is
    parameterised with `round_num` per run — the only phase that is
    instantiated anew per iteration.

    Input (in ctx):
      - ctx.research_plan (mandatory)
      - ctx.sources (for URL de-duplication)

    Output (in ctx):
      - ctx.sources (extended) — newly fetched documents

    fail_mode=CONTINUE: if a single search fails, the pipeline can carry
    on with what it has.
    """

    name = "search_and_fetch"
    fail_mode = FailMode.CONTINUE

    def __init__(self, orchestrator, round_num: int, progress_callback):
        self.orch = orchestrator
        self.round_num = round_num
        self.progress_callback = progress_callback

    def applies_to(self, ctx):
        if getattr(ctx, "research_plan", None) is None:
            return False, "no research_plan"
        return True, ""

    async def run(self, ctx):
        new_sources = await self.orch._run_search_and_fetch(
            ctx=ctx,
            round_num=self.round_num,
            progress_callback=self.progress_callback,
        )
        if new_sources:
            ctx.sources.extend(new_sources)

        # Classifier calls made inside the orchestrator (search_scope, the
        # off-topic filter) are appended to self.orch._classifier_calls, not
        # to ctx — they would otherwise be invisible in the pipeline-run tab.
        # Mirror the delta here (a high-water mark prevents duplicates
        # across rounds; like the _filter_stats_per_round mirror in
        # HarvestNode).
        try:
            allc = getattr(self.orch, "_classifier_calls", []) or []
            mirrored = getattr(self.orch, "_cc_mirrored", 0)
            if len(allc) > mirrored and hasattr(ctx, "classifier_calls"):
                ctx.classifier_calls.extend(allc[mirrored:])
                self.orch._cc_mirrored = len(allc)
        except Exception:
            pass  # visibility is nice to have, it must never break the pipeline

        return {
            "round": self.round_num,
            "new_sources": len(new_sources) if new_sources else 0,
            "total_sources": len(ctx.sources),
        }


# ── 10. HarvestNode ───────────────────────────────────────────────


class HarvestNode(PipelineNode):
    """Wrapper for `ResearchOrchestrator._run_harvest` (phase 3).

    Extracts facts from the fetched sources. Note: `_run_harvest` itself
    contains the query-anchor classifier and the person hallucination
    filter. This wrapper does not duplicate that — it simply hands over
    to the method; its behaviour is identical to that of the
    orchestrator's `run()` method.

    Input (in ctx):
      - ctx.research_plan (mandatory)
      - ctx.sources (the NEW sources of this round, provided by the
        caller — see `new_sources_filter`)

    Output (in ctx):
      - ctx.harvest_results (extended)
      - ctx.extracts (extended, positive only)
      - ctx.query_anchor (set by _run_harvest itself)
      - ctx.filter_stats_per_round (extended)
    """

    name = "harvest"
    fail_mode = FailMode.CONTINUE

    def __init__(
        self,
        orchestrator,
        new_sources: list,
        progress_callback,
    ):
        self.orch = orchestrator
        self.new_sources = new_sources
        self.progress_callback = progress_callback

    def applies_to(self, ctx):
        if not self.new_sources:
            return False, "no new sources to harvest"
        if getattr(ctx, "research_plan", None) is None:
            return False, "no research_plan"
        return True, ""

    async def run(self, ctx):
        # QueryAnchorNode sets ctx.query_anchor, but NOT the orchestrator
        # cache self.orch._query_anchor. _run_harvest checks
        # self._query_anchor, though, and would otherwise classify the anchor
        # a SECOND time (needless LLM call + duplicate entry in the pipeline
        # run). Bridge the two here, before _run_harvest runs.
        _qa = getattr(ctx, "query_anchor", None)
        if _qa is not None and getattr(self.orch, "_query_anchor", None) is None:
            self.orch._query_anchor = _qa

        results = await self.orch._run_harvest(
            sources=self.new_sources,
            plan=ctx.research_plan,
            progress_callback=self.progress_callback,
            query=ctx.query,
        )
        ctx.harvest_results.extend(results)

        # Collect extracts (positive only — as in run())
        new_extracts = []
        for hr in results:
            for e in hr.extracts:
                if getattr(e, "polarity", "positive") == "positive":
                    new_extracts.append(e)
        ctx.extracts.extend(new_extracts)

        # Mirror _filter_stats_per_round into ctx (as in run())
        if hasattr(self.orch, "_filter_stats_per_round"):
            ctx.filter_stats_per_round = list(self.orch._filter_stats_per_round)

        return {
            "harvest_results": len(results),
            "new_extracts": len(new_extracts),
        }


# ── 11. ContradictionCheckNode ────────────────────────────────────


class ContradictionCheckNode(PipelineNode):
    """Wrapper for `ResearchOrchestrator._check_contradictions` (phase 3c).

    Checks the collected extracts for contradictions in content.

    Input (in ctx):
      - ctx.extracts
      - ctx.research_plan

    Output (in ctx):
      - ctx.contradiction_warnings (set)
    """

    name = "contradiction_check"
    fail_mode = FailMode.CONTINUE

    def __init__(self, orchestrator, min_extracts: int = 4):
        self.orch = orchestrator
        self.min_extracts = min_extracts

    def applies_to(self, ctx):
        if len(getattr(ctx, "extracts", [])) < self.min_extracts:
            return False, f"fewer than {self.min_extracts} extracts"
        return True, ""

    async def run(self, ctx):
        contradictions = await self.orch._check_contradictions(ctx)
        ctx.contradiction_warnings = contradictions or []
        return {"contradictions": len(ctx.contradiction_warnings)}


# ── 12. SynthesisNode ─────────────────────────────────────────────


class SynthesisNode(PipelineNode):
    """Wrapper for `ResearchOrchestrator._run_synthesis` (phase 4).

    Creates the report. Without a report the pipeline is pointless —
    fail_mode=STOP.

    Input (in ctx):
      - ctx.research_plan
      - ctx.extracts
      - ctx.output_schema

    Output (in ctx):
      - ctx.final_report
    """

    name = "synthesis"
    fail_mode = FailMode.STOP

    def __init__(self, orchestrator, context_docs: str, progress_callback):
        self.orch = orchestrator
        self.context_docs = context_docs
        self.progress_callback = progress_callback

    def applies_to(self, ctx):
        if getattr(ctx, "research_plan", None) is None:
            return False, "no research_plan"
        return True, ""

    async def run(self, ctx):
        report = await self.orch._run_synthesis(
            ctx=ctx,
            context_docs=self.context_docs,
            progress_callback=self.progress_callback,
        )
        ctx.final_report = report or ""

        # Plausibility check: a report made of header + footer is technically
        # a string but contains nothing. Without this check the run would
        # count as successful and the Word export would go through.
        degenerate = len(ctx.final_report.strip()) < _MIN_REPORT_CHARS
        if degenerate:
            logger.error(
                "Synthesis returned only %d characters — report without content "
                "(extracts: %d, map answers: %d)",
                len(ctx.final_report.strip()),
                len(getattr(ctx, "extracts", []) or []),
                len(getattr(ctx, "map_answers", []) or []),
            )
            if not getattr(ctx, "synthesis_degraded", ""):
                ctx.synthesis_degraded = (
                    catalog_t("report.synthesis_empty", getattr(ctx, "output_language", None))
                )

        return {
            "report_length": len(ctx.final_report),
            "degenerate": degenerate,
        }


# ── 13. DiagnosisBannerNode ───────────────────────────────────────


class DiagnosisBannerNode(PipelineNode):
    """Append the diagnosis banner to the report.

    NO classifier call — string manipulation only. Reads
    `ctx.final_diagnosis` and annotates `ctx.final_report` accordingly.

    Input (in ctx):
      - ctx.final_report
      - ctx.final_diagnosis (may be None)

    Output (in ctx):
      - ctx.final_report (possibly with a banner)
    """

    name = "diagnosis_banner"
    fail_mode = FailMode.CONTINUE

    def applies_to(self, ctx):
        if not getattr(ctx, "final_report", "").strip():
            return False, "no report"
        diag = getattr(ctx, "final_diagnosis", None)
        if not diag or not diag.get("is_problematic"):
            return False, "diagnosis not problematic"
        return True, ""

    async def run(self, ctx):
        d = ctx.final_diagnosis
        banner = (
            catalog_t("report.diagnosis_banner", getattr(ctx, "output_language", None), diagnosis=d['diagnosis'], message=d.get('user_message', '').strip())
        )
        remediation = (d.get("remediation") or "").strip()
        if remediation:
            banner += catalog_t("report.recommendation_line", getattr(ctx, "output_language", None), text=remediation)
        ctx.final_report = ctx.final_report + banner
        return {"banner_appended": True}


# ── 14. ReportQualityNode ─────────────────────


class ReportQualityNode(PipelineNode):
    """Check the report for completeness, hallucinations and consistency.

    Complements the factoid verifier: the factoid verifier works
    atomically (statement by statement), the quality check at the macro
    level (the report as a whole).

    Mutations of ctx:
      - ctx.report_quality (set: QualityCheckResult dict)
    """

    name = "report_quality"
    fail_mode = FailMode.CONTINUE

    def __init__(self, llm):
        self.llm = llm

    def applies_to(self, ctx):
        if not getattr(ctx, "final_report", "").strip():
            return False, "no report"
        # Use-case override: a profile can explicitly disable the quality
        # check (e.g. for very fast research runs).
        return True, ""

    async def run(self, ctx):
        from src.pipeline.report_quality import check_report_quality
        result = await check_report_quality(
            report=ctx.final_report,
            extracts=getattr(ctx, "extracts", []),
            plan=getattr(ctx, "research_plan", None),
            dual_llm=self.llm,
        )
        ctx.report_quality = {
            "passed": result.passed,
            "rating": result.rating,
            "issues": [
                {
                    "art": m.art,
                    "description": m.description,
                    "suggestion": m.suggestion,
                    "question_id": m.question_id,
                    "segments": list(m.segments),
                }
                for m in result.issues
            ],
            "segments": result.segments,
            "fallback_used": result.fallback_used,
        }
        if not result.passed:
            logger.warning(
                "Quality check: report did NOT pass (%s) — %d findings",
                result.rating, len(result.issues),
            )
        return {
            "passed": result.passed,
            "issues": len(result.issues),
        }


# ── 15. QueryFulfillmentNode ──────────────────


class QueryFulfillmentNode(PipelineNode):
    """Check whether the report fulfils the original user request.

    Complements coverage, which checks per plan question, with an
    assessment at the level of the original query: not "was question F1
    answered?", but "was the request fulfilled *as a whole*?".

    Mutations of ctx:
      - ctx.query_fulfillment (set: FulfillmentResult dict)
    """

    name = "query_fulfillment"
    fail_mode = FailMode.CONTINUE

    def __init__(self, llm):
        self.llm = llm

    def applies_to(self, ctx):
        if not getattr(ctx, "final_report", "").strip():
            return False, "no report"
        if not getattr(ctx, "query", "").strip():
            return False, "no query"
        return True, ""

    async def run(self, ctx):
        from src.pipeline.report_quality import check_query_fulfillment
        result = await check_query_fulfillment(
            query=ctx.query,
            report=ctx.final_report,
            dual_llm=self.llm,
        )
        ctx.query_fulfillment = {
            "fulfilled": result.fulfilled,
            "assessment": result.assessment,
            "rework": result.rework,
            "fallback_used": result.fallback_used,
        }
        if not result.fulfilled:
            logger.warning(
                "Fulfilment check: request not fulfilled — %s",
                result.assessment[:100],
            )
        return {"fulfilled": result.fulfilled}


# ── 16. ReportWarningBannerNode ───────────────────────────────────


class ReportWarningBannerNode(PipelineNode):
    """Make negative check results visible in the report itself.

    Quality check and fulfilment check can correctly recognise an empty
    report — but if the result only appears in the server log, users get
    a neatly formatted Word document without content and without any
    hint that the pipeline already knew.

    This node makes no LLM call — a pure string operation over existing
    check results.

    Input (in ctx):
      - ctx.final_report
      - ctx.report_quality, ctx.query_fulfillment, ctx.synthesis_degraded

    Output (in ctx):
      - ctx.final_report (possibly with a warning block)
    """

    name = "report_warning_banner"
    fail_mode = FailMode.CONTINUE

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        if not getattr(ctx, "final_report", "").strip():
            return False, "no report"
        if self._problems(ctx):
            return True, ""
        return False, "no complaints"

    @staticmethod
    def _problems(ctx: Any) -> list[str]:
        problems: list[str] = []

        degraded = getattr(ctx, "synthesis_degraded", "") or ""
        if degraded:
            problems.append(degraded)

        quality = getattr(ctx, "report_quality", None) or {}
        if quality and not quality.get("passed", True):
            issues = quality.get("issues", []) or []
            assessment = quality.get("rating", "?")
            try:
                assessment = catalog_t(f"report.rating.{assessment}",
                                       getattr(ctx, "output_language", None))
            except KeyError:
                pass
            details = "; ".join(
                str(m.get("description", "")).strip()
                for m in issues[:3]
                if m.get("description")
            )
            text = catalog_t("report.quality_failed", getattr(ctx, "output_language", None), rating=assessment)
            if details:
                text += f": {details}"
            problems.append(text)

        fulfillment = getattr(ctx, "query_fulfillment", None) or {}
        if fulfillment and not fulfillment.get("fulfilled", True):
            text = catalog_t("report.request_not_fulfilled", getattr(ctx, "output_language", None))
            assessment = (fulfillment.get("assessment") or "").strip()
            if assessment:
                text += f": {assessment}"
            rework = (fulfillment.get("rework") or "").strip()
            if rework:
                text += f" — Empfehlung: {rework}"
            problems.append(text)

        return problems

    async def run(self, ctx: Any) -> dict:
        problems = self._problems(ctx)
        banner = (
            catalog_t("report.quality_banner", getattr(ctx, "output_language", None))
        )
        for p in problems:
            banner += f"- {p}\n"
        ctx.final_report = ctx.final_report + banner
        logger.warning(
            "Report delivered with a warning banner (%d finding(s))",
            len(problems),
        )
        return {"problems": len(problems)}
