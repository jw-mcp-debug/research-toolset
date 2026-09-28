"""
Central limits and constants of the research pipeline.

Numbers that would otherwise be scattered through the orchestrator and
other modules (maximum content size, maximum results, token limits,
various thresholds) are collected here. When adjusting them, change them
ONLY here.

Convention: constants in SCREAMING_SNAKE_CASE.
"""

# ─── Harvest / content sizes ──────────────────────────────────────

# Maximum number of characters per source that go into the harvest prompt.
# 80k characters ≈ 25k tokens, leaves room for prompt and output.
HARVEST_MAX_CONTENT_CHARS = 80_000

# Maximum tokens for the harvest answer (fact extraction per source).
HARVEST_MAX_OUTPUT_TOKENS = 8192


# ─── Search / search results ───────────────────────────────────────

# Default maximum hits per SearXNG query.
SEARCH_MAX_RESULTS_PER_QUERY = 50

# Maximum URLs actually fetched per research round,
# after de-duplication and the relevance filter.
FETCH_MAX_URLS_PER_ROUND = 100

# Maximum parallel fetches.
FETCH_MAX_PARALLEL = 8

# Maximum parallel harvest calls.
#
# NOTE: this value has NO effect — it is not imported anywhere.
# The default actually used is in `config.py`
# (`DualLLMConfig.from_env`, depending on whether the harvest and
# primary model are identical) and can be overridden via `HARVEST_MAX_PARALLEL`.
# Kept here so that existing imports do not break.
HARVEST_MAX_PARALLEL = 4


# ─── Research rounds ──────────────────────────────────────────────

# Default number of rounds (can be overridden per use case and CLI
# argument).
DEFAULT_MAX_ROUNDS = 3

# Hard maximum that is never exceeded, whatever the use case or CLI
# says — a safety cap against endless runs.
HARD_MAX_ROUNDS = 8


# ─── Classifier thresholds ───────────────────────────────────────

# Default confidence threshold above which a classifier verdict is
# considered "binding". Below this threshold the conservative default
# of the respective classifier applies.
#
# Concretely for query_anchor: the person filter is only activated
# at >= 0.7. Below that: no filter (safe default).
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.7

# Threshold below which the classifier itself falls back to its
# conservative default (e.g. evaluate_coverage → "partial"
# instead of "answered/unanswered").
CLASSIFIER_CONFIDENCE_FALLBACK = 0.5

# Filter loss rate at which diagnose_pipeline_state may decide
# "filter_too_strict".
FILTER_LOSS_RATE_WARNING = 0.50  # 50%
FILTER_LOSS_RATE_CRITICAL = 0.70  # 70%


# ─── Classifier batching ────────────────────────────────────────

# Items per batch for batch-capable classifiers
# (judge_source_relevance, verify_factoid_against_extracts).
CLASSIFIER_BATCH_SIZE = 5

# Maximum parallel classifier batches.
#
# Classifier calls go through `HarvestModelAdapter` → `harvest_complete`
# and thus through the `harvest_semaphore` (HARVEST_MAX_PARALLEL)
# anyway. As long as this value does not exceed that semaphore, it does
# not change the peak load on the harvest model — it only avoids a
# second, narrower throttle in front of it. With 100 sources (20
# batches), a limit of 4 would mean 5 waves instead of 2.
CLASSIFIER_MAX_PARALLEL = 10


# ─── Synthesis report ──────────────────────────────────────────────

# Maximum tokens for the synthesis output. Longer reports are
# processed segment by segment.
SYNTHESIS_MAX_TOKENS = 16_384

# If the report gets longer than the following, the final quality
# check is segmented.
SYNTHESIS_QC_SEGMENT_THRESHOLD_CHARS = 12_000


# ─── HTTP / network ───────────────────────────────────────────────

# Default timeout for fetches (seconds).
FETCH_TIMEOUT_SECONDS = 30.0

# Default timeout for LLM calls (seconds).
LLM_DEFAULT_TIMEOUT_SECONDS = 180.0
