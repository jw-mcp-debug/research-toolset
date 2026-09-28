"""
Adapter between DualLLMClient and the LLMCallable interface of the DAGExecutor.

The DAGExecutor expects a simple async function:
    async def llm_call(prompt: str, meta: dict) -> tuple[str, dict]

This module provides an adapter that, depending on
`meta["model_preference"]` (`primary`/`harvest`) and `meta["use_thinking"]`,
calls the right DualLLMClient endpoint and returns LLM metadata.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.client import DualLLMClient

logger = logging.getLogger(__name__)


def make_llm_callable(client: "DualLLMClient",
                      max_tokens_default: int = 4096):
    """Create an LLMCallable that uses the DualLLMClient.

    Returns:
        A function `async (prompt, meta) -> (output, llm_meta)`.
        meta fields used:
        - model_preference: "primary" | "harvest"
        - use_thinking: bool
        - max_tokens: int (optional)
        - task_id: str (for logging)
    """

    async def llm_call(prompt: str, meta: dict) -> tuple[str, dict]:
        task_id = meta.get("task_id", "?")
        model_pref = meta.get("model_preference", "harvest")
        use_thinking = meta.get("use_thinking", False)
        max_tokens = meta.get("max_tokens", max_tokens_default)

        messages = [{"role": "user", "content": prompt}]

        # Token counter before the call
        if model_pref == "primary":
            tokens_before = (
                client.primary.total_prompt_tokens
                + client.primary.total_completion_tokens
            )
            output = await client.primary_complete(
                messages,
                max_tokens=max_tokens,
                enable_thinking=use_thinking,
            )
            tokens_after = (
                client.primary.total_prompt_tokens
                + client.primary.total_completion_tokens
            )
        else:  # harvest
            tokens_before = (
                client.harvest.total_prompt_tokens
                + client.harvest.total_completion_tokens
            )
            output = await client.harvest_complete(
                messages, max_tokens=max_tokens,
            )
            tokens_after = (
                client.harvest.total_prompt_tokens
                + client.harvest.total_completion_tokens
            )

        tokens_used = tokens_after - tokens_before
        logger.debug(
            f"LLM call task={task_id} model={model_pref} "
            f"thinking={use_thinking} tokens={tokens_used}"
        )

        return output, {
            "tokens_used": tokens_used,
            "model": model_pref,
        }

    return llm_call


async def call_primary_json(client: "DualLLMClient", prompt: str,
                            max_tokens: int = 4096) -> dict:
    """Convenience: primary-LLM call with JSON output.

    Used by decomposers that produce structured plans.
    """
    messages = [{"role": "user", "content": prompt}]
    return await client.primary_complete_json(messages, max_tokens=max_tokens)
