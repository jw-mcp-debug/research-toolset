"""
Prompt package of the pipeline.

- research.py       — prompts of the research pipeline (web, institution,
                      literature check) and of the chat
- synthesis_map.py  — map phase of the synthesis (map + reduce)

The prompts of the analysis use cases are defined in
`src/pipeline/analysis_pipeline.py`.

All prompts are re-exported from `src.prompts`, so code can use
`from src.prompts import HARVEST_PROMPT`.
"""

# ── Re-export all research prompts ──
from src.prompts.research import (  # noqa: F401
    get_date_text,
    SYSTEM_PROMPT_CHAT,
    FORMAT_AGENT_PROMPT,
    ANALYSIS_PROMPT,
    INSTITUTION_ANALYSIS_PROMPT,
    HARVEST_PROMPT,
    SYNTHESIS_PROMPT,
    REPORT_REVISION_PROMPT,
    SUMMARY_PROMPT,
    CONTRADICTION_PROMPT,
)

# ── Synthesis: map phase (map + reduce) ──
from src.prompts.synthesis_map import (  # noqa: F401
    QUESTION_ANSWER_PROMPT,
)

__all__ = [
    "get_date_text",
    "SYSTEM_PROMPT_CHAT",
    "FORMAT_AGENT_PROMPT",
    "ANALYSIS_PROMPT",
    "INSTITUTION_ANALYSIS_PROMPT",
    "HARVEST_PROMPT",
    "SYNTHESIS_PROMPT",
    "REPORT_REVISION_PROMPT",
    "QUESTION_ANSWER_PROMPT",
    "SUMMARY_PROMPT",
    "CONTRADICTION_PROMPT",
]
