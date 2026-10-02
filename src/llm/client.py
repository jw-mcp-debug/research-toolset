"""
LLM client with support for two models.
Primary (complex): analysis, synthesis, format agent
Harvest (simple): fact extraction, in parallel
"""

import asyncio
import json
import logging
import re
from typing import AsyncGenerator

import httpx
from openai import AsyncOpenAI

from src.config import LLMConfig, DualLLMConfig

logger = logging.getLogger(__name__)


# The placeholder that `_process_think_tags` emits while a <think> block
# is still open. It is display, never content — hence defined once
# for all places that emit or remove it.
THINKING_PLACEHOLDER = "🤔 *Thinking...*"
# Recognises the placeholder in any wording (the emoji and the italic
# span are the fixed part), so changing or translating the text above
# cannot break the clean-up.
_PLACEHOLDER_RE = re.compile(r"🤔\s*\*[^*\n]{1,40}\*")


class EmptyLLMResponseError(RuntimeError):
    """The LLM returned no visible answer.

    Typical cause with reasoning models (e.g. Kimi): the token budget
    (`max_tokens`) is used up entirely by reasoning, `finish_reason` is
    "length" and the `content` field stays empty — the actual answer was
    never generated.

    Important: Kimi reasoning can NOT be switched off by a flag (there
    is no `enable_thinking` equivalent). The only effective remedies are
    therefore a sufficient token budget and a fallback in the caller.
    That is why this is an explicit exception and not a silent empty
    string: the caller MUST decide what to do.
    """

    def __init__(self, message: str, *, finish_reason: str | None = None,
                 reasoning_chars: int = 0):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.reasoning_chars = reasoning_chars


def _reasoning_of(obj) -> str:
    """Read the reasoning text from a message or delta object.

    Reasoning models do NOT deliver their reasoning in `content` but in
    an additional field. The field names differ by server/model
    (`reasoning_content` with vLLM/Kimi/DeepSeek, `reasoning` with some
    gateways) — hence both are checked.

    Without reading it, the client sees only empty chunks during the
    whole reasoning phase and cannot tell "model is still thinking" from
    "model delivers nothing".
    """
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(obj, attr, None)
        if val:
            return str(val)
    return ""


def _repair_truncated_json(text: str) -> dict | None:
    """Try to repair truncated JSON.

    Strategy: try the fields one after another with json.loads; on error
    cut off the incomplete value and close the object. Works for the
    most common pattern:
    {"key1": value1, "key2": value2, "key3": "trunc...
    """
    text = text.strip()
    if not text.startswith("{"):
        return None

    # Try appending progressively more closing characters
    for suffix in ["}", '"}', '"]}', '"}]}',"]}}", '""}}']:
        try:
            result = json.loads(text + suffix)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    # More aggressive: cut from the end until the JSON is valid
    for cut in range(1, min(200, len(text))):
        candidate = text[:-cut].rstrip()
        # Look for the last complete field
        for suffix in ["}", '"}', '"]}', '"}]}']:
            # find the last comma, cut after it
            last_comma = candidate.rfind(",")
            if last_comma > 0:
                try:
                    result = json.loads(candidate[:last_comma] + suffix)
                    if isinstance(result, dict) and result:
                        return result
                except json.JSONDecodeError:
                    continue

    return None


class LLMClient:
    """Single LLM client with streaming and non-streaming calls."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
            max_retries=3,
            timeout=httpx.Timeout(
                connect=15.0,
                read=config.timeout,
                write=30.0,
                pool=15.0,
            ),
        )
        self._stop_event = asyncio.Event()
        # Token-Tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

        # Thinking mode: send enable_thinking per request to vLLM;
        # overrides the server default (chat-template kwargs)
        self._enable_thinking = getattr(config, "enable_thinking", None)
        if self._enable_thinking is None:
            import os
            val = os.environ.get("LLM_ENABLE_THINKING", "true").lower()
            self._enable_thinking = val in ("true", "1", "yes", "on")

        if self._enable_thinking:
            logger.info(
                f"LLM {config.model_name}: thinking mode enabled "
                f"(enable_thinking=true per Request)"
            )

    def _apply_thinking(self, kwargs: dict):
        """Set enable_thinking in extra_body for vLLM."""
        if self._enable_thinking:
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["chat_template_kwargs"] = {
                "enable_thinking": True
            }

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Non-streaming completion with retry on 503/429.

        Args:
            enable_thinking: overrides the global thinking setting.
                True = force thinking, False = disable thinking,
                None = use the global setting.
        """
        max_tok = max_tokens or self.config.max_output_tokens

        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_tok,
        }

        if not self.config.is_kimi():
            kwargs["temperature"] = temperature or self.config.default_temperature
            kwargs["top_p"] = self.config.default_top_p

        # Thinking: explicit override or global setting
        if enable_thinking is True:
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["chat_template_kwargs"] = {
                "enable_thinking": True
            }
        elif enable_thinking is False:
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["chat_template_kwargs"] = {
                "enable_thinking": False
            }
        else:
            self._apply_thinking(kwargs)

        max_retries = 3
        base_delay = 5.0  # seconds
        budget_escalated = False  # only once per call

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                # Token-Tracking
                self.total_requests += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens or 0
                    self.total_completion_tokens += response.usage.completion_tokens or 0
                choice = response.choices[0]
                content = choice.message.content or ""
                visible = self._strip_think_tags(content)

                if visible:
                    return visible

                # ── Empty visible answer: determine the cause ──
                reasoning = _reasoning_of(choice.message)
                finish_reason = getattr(choice, "finish_reason", None)

                if reasoning or finish_reason == "length":
                    # Reasoning ate the budget. Without this, the synthesis
                    # comes back empty — so retry once with a larger budget
                    # instead of returning "".
                    logger.warning(
                        "LLM %s: no visible answer — %d characters of "
                        "reasoning, finish_reason=%s, max_tokens=%d",
                        self.config.model_name, len(reasoning),
                        finish_reason, max_tok,
                    )
                    room = self.config.max_output_tokens
                    if (not budget_escalated and finish_reason == "length"
                            and max_tok < room):
                        max_tok = min(max_tok * 4, room)
                        kwargs["max_tokens"] = max_tok
                        budget_escalated = True
                        logger.info(
                            "LLM %s: retrying with max_tokens=%d "
                            "(reasoning budget exhausted)",
                            self.config.model_name, max_tok,
                        )
                        continue

                    raise EmptyLLMResponseError(
                        f"{self.config.model_name}: empty answer "
                        f"(finish_reason={finish_reason}, "
                        f"{len(reasoning)} characters of reasoning, "
                        f"max_tokens={max_tok})",
                        finish_reason=finish_reason,
                        reasoning_chars=len(reasoning),
                    )

                # No reasoning, no truncation → the model deliberately said
                # nothing. Return it as it is.
                return visible
            except EmptyLLMResponseError:
                raise
            except Exception as e:
                error_str = str(e)
                is_retryable = (
                    "503" in error_str
                    or "429" in error_str
                    or "502" in error_str
                    or "Service Temporarily Unavailable" in error_str
                    or "overloaded" in error_str.lower()
                )

                if is_retryable and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 5s, 10s, 20s
                    logger.warning(
                        f"LLM {self.config.model_name}: Retry {attempt + 1}/{max_retries} "
                        f"after {delay:.0f}s (error: {error_str[:100]})"
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error(f"LLM complete error: {e}")
                raise

    async def complete_json(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> dict:
        """Completion with JSON parsing and retry on error."""
        # Classifiers expect {} here rather than an exception — they have
        # their own fallbacks. The cause is logged anyway so that it can
        # be found in the log.
        try:
            raw = await self.complete(messages, max_tokens)
        except EmptyLLMResponseError as e:
            logger.warning("JSON completion without a visible answer: %s", e)
            return {}
        result = self._parse_json(raw)

        if result:
            return result

        # Retry: append an explicit JSON instruction
        logger.info("JSON parse failed, retrying with an explicit instruction")
        retry_messages = list(messages) + [
            {"role": "assistant", "content": raw[:500] if raw else ""},
            {"role": "user", "content": (
                "Your answer could not be parsed as JSON. "
                "Please answer ONLY with a valid JSON object, "
                "without ``` code blocks, without explanations, without <think> tags. "
                "Start directly with { and end with }."
            )},
        ]
        try:
            raw2 = await self.complete(retry_messages, max_tokens)
        except EmptyLLMResponseError as e:
            logger.warning("JSON retry without a visible answer: %s", e)
            return {}
        result2 = self._parse_json(raw2)

        if result2:
            logger.info("JSON retry successful")
            return result2

        logger.error("JSON retry failed as well")
        return {}

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming generator with think-tag filtering and retry.

        Yields only visible text. Reasoning chunks are counted silently
        and not emitted. If the stream ends WITHOUT any visible text,
        `EmptyLLMResponseError` is raised. The caller must therefore
        never take the last emitted chunk as the result unchecked.

        Args:
            enable_thinking: as for `complete()`. Note: the switch only
                works for models with `enable_thinking` in their chat
                template (Qwen family). Kimi reasoning can NOT be switched
                off this way — only a sufficient budget helps there.
        """
        self._stop_event.clear()
        max_tok = max_tokens or self.config.max_output_tokens

        kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_tok,
            "stream": True,
        }

        if not self.config.is_kimi():
            kwargs["temperature"] = temperature or self.config.default_temperature
            kwargs["top_p"] = self.config.default_top_p

        if enable_thinking is not None:
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"]["chat_template_kwargs"] = {
                "enable_thinking": bool(enable_thinking)
            }
        else:
            self._apply_thinking(kwargs)

        max_retries = 3
        budget_escalated = False
        for attempt in range(max_retries):
            try:
                stream_resp = await self.client.chat.completions.create(**kwargs)

                full_response = ""
                thinking = False
                visible_response = ""
                got_content = False
                reasoning_response = ""
                finish_reason = None
                stopped = False

                async for chunk in stream_resp:
                    if self._stop_event.is_set():
                        try:
                            await stream_resp.close()
                        except Exception:
                            pass
                        stopped = True
                        break

                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason

                    delta = choice.delta

                    # Reasoning chunks (Kimi & co. deliver them in a separate
                    # field, NOT in `content`). They are only counted, not
                    # emitted: the amount is the diagnosis if nothing visible
                    # comes at the end. Deliberately NO yield here — one UI
                    # update per reasoning chunk would mean tens of thousands
                    # of updates.
                    reasoning = _reasoning_of(delta)
                    if reasoning:
                        reasoning_response += reasoning
                        continue

                    content = delta.content if delta.content else ""
                    if not content:
                        continue

                    got_content = True
                    full_response += content

                    result = self._process_think_tags(
                        full_response, thinking, visible_response
                    )
                    thinking = result["thinking"]
                    visible_response = result["visible"]

                    yield visible_response

                if stopped:
                    return

                # Stream completed successfully
                self.total_requests += 1
                # Estimate: ~4 characters per token for the output
                # (reasoning counts too — it costs real budget)
                self.total_completion_tokens += (
                    len(full_response) + len(reasoning_response)
                ) // 4

                # ── Guard: a stream must have produced visible text ──
                # A stream that only delivers reasoning ends silently; without
                # this check the caller would take the placeholder (or "") as
                # the finished report.
                #
                # The placeholder from `_process_think_tags` does NOT count as
                # visible text here — otherwise a stream with an unclosed <think>
                # tag would go undetected.
                visible_clean = _PLACEHOLDER_RE.sub("", visible_response)
                if not (got_content and visible_clean.strip()):
                    logger.warning(
                        "LLM %s: stream without visible text — %d characters of "
                        "reasoning, finish_reason=%s, max_tokens=%d",
                        self.config.model_name, len(reasoning_response),
                        finish_reason, max_tok,
                    )
                    room = self.config.max_output_tokens
                    if (not budget_escalated
                            and finish_reason == "length"
                            and max_tok < room):
                        max_tok = min(max_tok * 4, room)
                        kwargs["max_tokens"] = max_tok
                        budget_escalated = True
                        logger.info(
                            "LLM %s: retrying the stream with max_tokens=%d "
                            "(reasoning budget exhausted)",
                            self.config.model_name, max_tok,
                        )
                        continue
                    if attempt < max_retries - 1:
                        continue
                    raise EmptyLLMResponseError(
                        f"{self.config.model_name}: stream delivered no "
                        f"visible text (finish_reason={finish_reason}, "
                        f"{len(reasoning_response)} characters of reasoning, "
                        f"max_tokens={max_tok})",
                        finish_reason=finish_reason,
                        reasoning_chars=len(reasoning_response),
                    )
                return

            except EmptyLLMResponseError:
                raise
            except asyncio.CancelledError:
                raise
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError,
                    ConnectionError, OSError) as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 2
                    logger.warning(
                        f"LLM stream connection failed (attempt {attempt + 1}/"
                        f"{max_retries}), Retry in {wait}s: {e}"
                    )
                    yield f"🔄 *Connection interrupted, retrying in {wait}s...*"
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"LLM stream failed after {max_retries} attempts: {e}")
                    yield f"⚠️ Connection error after {max_retries} attempts: {e}"
            except Exception as e:
                error_str = str(e)
                is_retryable = (
                    "503" in error_str or "429" in error_str
                    or "502" in error_str
                    or "Service Temporarily Unavailable" in error_str
                    or "overloaded" in error_str.lower()
                )
                if is_retryable and attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    logger.warning(
                        f"LLM Stream: Retry {attempt + 1}/{max_retries} "
                        f"after {wait}s (error: {error_str[:100]})"
                    )
                    yield f"🔄 *Server overloaded, retrying in {wait}s...*"
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"LLM stream error: {e}")
                    yield f"⚠️ Error: {e}"
                    return

    def stop(self):
        self._stop_event.set()

    # ─── Helper functions ────────────────────────────────────────────

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> from text.

        Also handles:
        - unclosed <think> tags (e.g. at the token limit)
        - several think blocks
        """
        # Remove closed tags
        result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Unclosed tags: remove everything from <think> to the end
        result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)
        return result.strip()

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from an LLM answer (also from a Markdown code block).

        Robust fallbacks for various LLM output formats.
        """
        clean = text.strip()

        if not clean:
            logger.warning("Empty LLM answer for JSON parsing")
            return {}

        # Attempt 1: parse directly
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract JSON from ```json ... ```
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Attempt 3: extract from the first { to the last }
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(clean[start:end + 1])
            except json.JSONDecodeError:
                # Attempt 3b: count nested braces
                depth = 0
                json_end = -1
                for i in range(start, len(clean)):
                    if clean[i] == "{":
                        depth += 1
                    elif clean[i] == "}":
                        depth -= 1
                        if depth == 0:
                            json_end = i
                            break
                if json_end > start:
                    try:
                        return json.loads(clean[start:json_end + 1])
                    except json.JSONDecodeError:
                        pass

        # Attempt 4: remove think tags again (in case they are inside JSON)
        no_think = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
        no_think = re.sub(r"<think>.*$", "", no_think, flags=re.DOTALL).strip()
        if no_think != clean and no_think:
            start = no_think.find("{")
            end = no_think.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(no_think[start:end + 1])
                except json.JSONDecodeError:
                    pass

        logger.warning(
            f"Could not parse JSON "
            f"(length={len(clean)}, "
            f"has_braces={'{' in clean}, "
            f"has_think={'<think>' in clean}): "
            f"{clean[:600]}"
        )

        # Attempt 5: repair truncated JSON
        # LLM output is sometimes cut off at max_tokens.
        # We try to rescue the top-level fields parsed so far.
        start = clean.find("{")
        if start >= 0:
            truncated = clean[start:]
            try:
                repaired = _repair_truncated_json(truncated)
                if repaired:
                    logger.info(
                        f"Truncated JSON repaired "
                        f"(keys: {list(repaired.keys())})"
                    )
                    return repaired
            except Exception:
                pass

        return {}

    @staticmethod
    def _process_think_tags(full_text: str, was_thinking: bool,
                            prev_visible: str) -> dict:
        """Think-tag processing for streaming."""
        open_tags = full_text.count("<think>")
        close_tags = full_text.count("</think>")
        currently_thinking = open_tags > close_tags

        if currently_thinking:
            return {"thinking": True, "visible": THINKING_PLACEHOLDER}

        # Detect a partial <think> tag
        if open_tags == 0:
            think_prefix = "<think>"
            stripped = full_text.lstrip()
            for i in range(1, len(think_prefix)):
                if stripped == think_prefix[:i]:
                    return {"thinking": True,
                            "visible": prev_visible or THINKING_PLACEHOLDER}

        # Remove think blocks
        visible = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)

        if "<think>" in visible and "</think>" not in visible.split("<think>")[-1]:
            visible = visible[:visible.rfind("<think>")]

        return {"thinking": False, "visible": visible.strip()}


class DualLLMClient:
    """Two LLM clients: primary (complex) + harvest (simple/fast)."""

    def __init__(self, config: DualLLMConfig):
        self.config = config
        self.primary = LLMClient(config.primary)
        self.harvest = LLMClient(config.harvest)
        self.harvest_semaphore = asyncio.Semaphore(config.harvest_max_parallel)

        # Harvest LLM: disable thinking (saves tokens)
        import os
        harvest_thinking = os.environ.get(
            "HARVEST_LLM_ENABLE_THINKING", "false"
        ).lower()
        self.harvest._enable_thinking = harvest_thinking in (
            "true", "1", "yes", "on"
        )
        if not self.harvest._enable_thinking:
            logger.info(
                f"LLM {config.harvest.model_name}: thinking mode "
                f"disabled (harvest — saves tokens)"
            )

    async def primary_complete(self, messages: list[dict],
                               max_tokens: int | None = None,
                               enable_thinking: bool | None = None) -> str:
        return await self.primary.complete(
            messages, max_tokens, enable_thinking=enable_thinking,
        )

    async def primary_complete_json(self, messages: list[dict],
                                    max_tokens: int | None = None) -> dict:
        return await self.primary.complete_json(messages, max_tokens)

    async def primary_stream(
        self, messages: list[dict],
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.primary.stream(
            messages, max_tokens, enable_thinking=enable_thinking,
        ):
            yield chunk

    async def harvest_complete(self, messages: list[dict],
                               max_tokens: int | None = None) -> str:
        """Harvest completion with a semaphore for parallelism.

        Returns "" instead of an exception for an answer that is empty
        because of reasoning: harvest and classifiers treat the empty
        string as "nothing found" and have their own fallbacks. A single
        extraction call must not abort the research.
        """
        async with self.harvest_semaphore:
            try:
                return await self.harvest.complete(messages, max_tokens)
            except EmptyLLMResponseError as e:
                logger.warning("Harvest without a visible answer: %s", e)
                return ""

    async def harvest_complete_json(self, messages: list[dict],
                                    max_tokens: int | None = None) -> dict:
        """Harvest completion with JSON parsing.

        Counterpart of `primary_complete_json`. Exists so that the choice
        of output format is independent of the choice of model.
        """
        async with self.harvest_semaphore:
            return await self.harvest.complete_json(messages, max_tokens)

    def stop(self):
        self.primary.stop()
        self.harvest.stop()

    def get_usage_stats(self) -> dict:
        """Return token usage and request counters of both LLMs."""
        return {
            "primary": {
                "model": self.config.primary.model_name,
                "requests": self.primary.total_requests,
                "prompt_tokens": self.primary.total_prompt_tokens,
                "completion_tokens": self.primary.total_completion_tokens,
                "total_tokens": (self.primary.total_prompt_tokens
                                 + self.primary.total_completion_tokens),
            },
            "harvest": {
                "model": self.config.harvest.model_name,
                "requests": self.harvest.total_requests,
                "prompt_tokens": self.harvest.total_prompt_tokens,
                "completion_tokens": self.harvest.total_completion_tokens,
                "total_tokens": (self.harvest.total_prompt_tokens
                                 + self.harvest.total_completion_tokens),
            },
            "total_requests": (self.primary.total_requests
                               + self.harvest.total_requests),
            "total_tokens": (self.primary.total_prompt_tokens
                             + self.primary.total_completion_tokens
                             + self.harvest.total_prompt_tokens
                             + self.harvest.total_completion_tokens),
        }

    def reset_usage(self):
        """Reset the token counters (before a new research run)."""
        for client in (self.primary, self.harvest):
            client.total_prompt_tokens = 0
            client.total_completion_tokens = 0
            client.total_requests = 0
