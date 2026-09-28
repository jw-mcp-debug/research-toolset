"""
LLM classifiers — they replace meaning-based heuristics.

Overview:
  - query_anchor.py        — detect a person/organisation anchor (hard blocker)
  - coverage.py            — coverage assessment and continue decision
  - diagnose.py            — final diagnosis of the pipeline run
  - source_relevance.py    — pre-fetch relevance filter (batched)
  - factoids.py            — factoid extraction and verification
  - base.py                — shared infrastructure

All classifiers follow the same pattern:
  - JSON-mode call with a structured schema
  - confidence field (0.0-1.0) with a conservative default at low confidence
  - reasoning field for debugging and display in the UI
  - structured logging (`ClassifierCall`)

Guiding principle: where a decision depends on MEANING, an LLM call is
used, not a regex or a score heuristic. Where it is purely mechanical,
it stays deterministic.
"""

from src.pipeline.classifiers.base import (
    ClassifierCall,
    ClassifierLLM,
    ClassifierResult,
    call_classifier,
    confidence_below_threshold,
    hash_prompt,
)
from src.pipeline.classifiers.coverage import (
    ContinueDecision,
    CoverageResult,
    decide_continue_research,
    evaluate_coverage,
)
from src.pipeline.classifiers.diagnose import (
    PROBLEMATIC_DIAGNOSES,
    DiagnosisResult,
    diagnose_pipeline_state,
)
from src.pipeline.classifiers.factoids import (
    Factoid,
    FactoidVerification,
    extract_factoids,
    verify_factoids_against_extracts,
)
from src.pipeline.classifiers.query_anchor import (
    QueryAnchor,
    classify_query_anchor,
)
from src.pipeline.classifiers.source_relevance import (
    SourceRelevance,
    judge_source_relevance,
)

__all__ = [
    # Basis
    "ClassifierCall",
    "ClassifierLLM",
    "ClassifierResult",
    "call_classifier",
    "confidence_below_threshold",
    "hash_prompt",
    # Query anchor
    "QueryAnchor",
    "classify_query_anchor",
    # Coverage
    "CoverageResult",
    "ContinueDecision",
    "evaluate_coverage",
    "decide_continue_research",
    # Diagnose
    "DiagnosisResult",
    "PROBLEMATIC_DIAGNOSES",
    "diagnose_pipeline_state",
    # Source-Relevance
    "SourceRelevance",
    "judge_source_relevance",
    # Factoids
    "Factoid",
    "FactoidVerification",
    "extract_factoids",
    "verify_factoids_against_extracts",
]
