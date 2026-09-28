"""
DAG executor: runs a TaskPlan in topologically sorted layers.

Design principles:
1. execution strictly layer by layer (no free DAG traversal)
2. strict state machine: pending → running → done | failed | skipped
3. single source of truth: ctx.task_results
4. declarative retries (SubTask.max_retries)
5. persistence after each layer via a callback
6. resuming from a persisted state is possible
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional

from src.pipeline.models import (
    HarvestContext,
    SubTask,
    TaskPlan,
    TaskResult,
    TaskState,
)
from src.utils.task_rate_limiter import TaskRateLimiter

logger = logging.getLogger(__name__)


# Type-Aliases
PromptDict = dict[str, str]                       # template_name -> prompt_template
ProgressCallback = Callable[[str, dict], Awaitable[None]]
PersistCallback = Callable[[HarvestContext, int], Awaitable[None]]
LLMCallable = Callable[[str, dict], Awaitable[tuple[str, dict]]]
# LLMCallable: (prompt, sub_task_meta) -> (output_text, llm_metadata)


class DAGExecutor:
    """Run a TaskPlan in topologically sorted layers.

    Usage:
        executor = DAGExecutor(
            llm_call=my_llm_caller,
            max_parallel=5,
            rate_limit=3.0,
        )
        results = await executor.execute(
            plan=task_plan,
            prompts={"my_template": "..."},
            context=harvest_ctx,
        )

    Layer-by-layer strategy:
    1. plan.topological_order() returns a list of layers
    2. per layer: in parallel, limited by a semaphore (max_parallel)
    3. per call: rate limiter with exponential back-off
    4. after each layer: persist ctx.task_results
    5. on errors: tasks with max_retries > 0 are retried
    6. tasks whose dependencies FAILED → SKIPPED
    7. resume: with resume_from_layer > 0, earlier layers are assumed
       to be done (from ctx.task_results)
    """

    def __init__(
        self,
        llm_call: LLMCallable,
        max_parallel: int = 5,
        rate_limit: float = 3.0,
        burst: int = 5,
        progress_callback: Optional[ProgressCallback] = None,
        persist_callback: Optional[PersistCallback] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        if max_parallel < 1 or max_parallel > 10:
            raise ValueError("max_parallel must be between 1 and 10")

        self.llm_call = llm_call
        self.max_parallel = max_parallel
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.rate_limiter = TaskRateLimiter(rate=rate_limit, burst=burst)
        self.progress_callback = progress_callback
        self.persist_callback = persist_callback
        self.stop_check = stop_check or (lambda: False)

    async def execute(
        self,
        plan: TaskPlan,
        prompts: PromptDict,
        context: HarvestContext,
        resume_from_layer: int = 0,
    ) -> dict[str, TaskResult]:
        """Run the plan.

        Args:
            plan: TaskPlan with sub-tasks and dependencies
            prompts: mapping from template names to prompt strings
            context: HarvestContext (modified: task_results, completed_layers)
            resume_from_layer: with > 0, layers 0..resume_from_layer-1 are
                              skipped (resume from a checkpoint)

        Returns:
            dict {task_id: TaskResult} (also in ctx.task_results)
        """
        layers = plan.topological_order()
        logger.info(
            f"DAGExecutor: {len(plan.tasks)} tasks in {len(layers)} "
            f"layers, max_parallel={self.max_parallel}"
        )

        # Init: set all tasks to PENDING (except when resuming)
        if resume_from_layer == 0:
            context.task_results = {
                task.id: TaskResult(task_id=task.id, state=TaskState.PENDING)
                for task in plan.tasks
            }
            context.completed_layers = []
        else:
            logger.info(f"Resume from layer {resume_from_layer}")

        # Run layer by layer
        for layer_idx, layer in enumerate(layers):
            if layer_idx < resume_from_layer:
                logger.debug(f"Layer {layer_idx}: skipped (resume)")
                continue

            # Abort check before every layer
            if self.stop_check():
                logger.info(
                    f"Stop requested before layer {layer_idx + 1} — "
                    f"remaining tasks are skipped"
                )
                # Mark the remaining tasks as SKIPPED
                for remaining_layer in layers[layer_idx:]:
                    for task in remaining_layer:
                        result = context.task_results.get(task.id)
                        if result and result.state == TaskState.PENDING:
                            result.state = TaskState.SKIPPED
                            result.error = "cancelled by the user"
                            result.started_at = datetime.now().isoformat()
                            result.finished_at = result.started_at
                await self._notify_progress("stop_requested", {
                    "layer": layer_idx + 1,
                })
                break

            await self._notify_progress("layer_start", {
                "layer": layer_idx + 1,
                "total_layers": len(layers),
                "tasks": [t.id for t in layer],
            })

            # Step 1: mark tasks with failed dependencies as skipped
            executable_tasks = []
            for task in layer:
                if self._dependencies_failed(task, context):
                    self._mark_skipped(task, context)
                    await self._notify_progress("task_skipped", {
                        "task_id": task.id,
                        "reason": "dependency failed",
                    })
                else:
                    executable_tasks.append(task)

            # Step 2: run the remaining tasks in parallel
            if executable_tasks:
                await self._execute_layer(executable_tasks, prompts, context)

            # Step 3: layer complete — persist
            context.completed_layers.append(layer_idx)
            if self.persist_callback:
                try:
                    await self.persist_callback(context, layer_idx)
                except Exception as e:
                    logger.error(f"Persist callback failed: {e}")

            # Step 4: layer statistics
            done = sum(
                1 for t in layer
                if context.task_results[t.id].state == TaskState.DONE
            )
            failed = sum(
                1 for t in layer
                if context.task_results[t.id].state == TaskState.FAILED
            )
            skipped = sum(
                1 for t in layer
                if context.task_results[t.id].state == TaskState.SKIPPED
            )
            logger.info(
                f"Layer {layer_idx + 1}/{len(layers)} completed: "
                f"{done} done, {failed} failed, {skipped} skipped"
            )
            await self._notify_progress("layer_done", {
                "layer": layer_idx + 1,
                "done": done,
                "failed": failed,
                "skipped": skipped,
            })

        return context.task_results

    async def _execute_layer(
        self,
        tasks: list[SubTask],
        prompts: PromptDict,
        context: HarvestContext,
    ) -> None:
        """Run all tasks of a layer in parallel."""
        async def run_one(task: SubTask) -> None:
            async with self.semaphore:
                await self._execute_task(task, prompts, context)

        await asyncio.gather(
            *(run_one(task) for task in tasks),
            return_exceptions=False,
        )

    async def _execute_task(
        self,
        task: SubTask,
        prompts: PromptDict,
        context: HarvestContext,
    ) -> None:
        """Run a single sub-task with retry logic."""
        result = context.task_results[task.id]
        result.state = TaskState.RUNNING
        result.started_at = datetime.now().isoformat()

        await self._notify_progress("task_start", {
            "task_id": task.id,
            "phase": task.phase,
            "description": task.description,
        })

        # Assemble the prompt
        try:
            prompt = self._build_prompt(task, prompts, context)
        except KeyError as e:
            result.state = TaskState.FAILED
            result.error = f"prompt template '{task.prompt_template}' not found: {e}"
            result.finished_at = datetime.now().isoformat()
            logger.error(f"Task {task.id}: {result.error}")
            await self._notify_progress("task_failed", {
                "task_id": task.id, "error": result.error,
            })
            return
        except Exception as e:
            result.state = TaskState.FAILED
            result.error = f"prompt build error: {e}"
            result.finished_at = datetime.now().isoformat()
            logger.error(f"Task {task.id}: {result.error}")
            await self._notify_progress("task_failed", {
                "task_id": task.id, "error": result.error,
            })
            return

        # Retry-Loop
        last_error = ""
        for attempt in range(task.max_retries + 1):
            try:
                await self.rate_limiter.acquire()
                output, meta = await self.llm_call(prompt, {
                    "task_id": task.id,
                    "phase": task.phase,
                    "model_preference": task.model_preference,
                    "use_thinking": task.use_thinking,
                })
                # success
                self.rate_limiter.report_success()
                result.output = output
                result.parsed_output = self._try_parse_json(output)
                result.tokens_used = meta.get("tokens_used", 0)
                result.retries_used = attempt
                result.state = TaskState.DONE
                result.finished_at = datetime.now().isoformat()
                await self._notify_progress("task_done", {
                    "task_id": task.id,
                    "tokens": result.tokens_used,
                    "retries": attempt,
                })
                return

            except RateLimitException as e:
                await self.rate_limiter.report_rate_limited()
                last_error = f"Rate-Limited: {e}"
                logger.warning(
                    f"Task {task.id} attempt {attempt + 1}: {last_error}"
                )
                # Retrying on rate limits beyond max_retries is not intended —
                # we wait and try again only within max_retries
                if attempt >= task.max_retries:
                    break

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    f"Task {task.id} attempt {attempt + 1}/{task.max_retries + 1}: "
                    f"{last_error}"
                )
                if attempt >= task.max_retries:
                    break
                # Short back-off between retries
                await asyncio.sleep(1.0 * (attempt + 1))

        # All retries used up
        result.state = TaskState.FAILED
        result.error = last_error
        result.retries_used = task.max_retries
        result.finished_at = datetime.now().isoformat()
        logger.error(f"Task {task.id} failed permanently: {last_error}")
        await self._notify_progress("task_failed", {
            "task_id": task.id, "error": last_error,
        })

    def _build_prompt(
        self,
        task: SubTask,
        prompts: PromptDict,
        context: HarvestContext,
    ) -> str:
        """Build the prompt string for a sub-task.

        Uses the template from prompts[task.prompt_template] and fills it
        with:
        1. task.prompt_params (static, at planning time)
        2. task.param_deps (template variable → task_id, resolved at run
           time from the dependencies)
        3. dep_<id> keys as a fallback for direct references
        4. running_context from ctx (if the template uses it)
        """
        if task.prompt_template not in prompts:
            raise KeyError(task.prompt_template)

        template = prompts[task.prompt_template]
        params = dict(task.prompt_params)

        # Resolve param_deps: template_var → task_id → output
        for template_var, dep_task_id in task.param_deps.items():
            dep_result = context.task_results.get(dep_task_id)
            if dep_result and dep_result.state == TaskState.DONE:
                params[template_var] = dep_result.output
            else:
                params[template_var] = "(not available)"

        # concat_param_deps: template_var → list[task_id] → concatenated
        for template_var, dep_task_ids in task.concat_param_deps.items():
            blocks = []
            for tid in dep_task_ids:
                dep_result = context.task_results.get(tid)
                if dep_result and dep_result.state == TaskState.DONE:
                    blocks.append(f"## {tid}\n\n{dep_result.output}")
            params[template_var] = "\n\n---\n\n".join(blocks) or "(empty)"

        # dep_<id> keys for direct references
        for dep_id in task.depends_on:
            safe_key = f"dep_{dep_id}".replace(":", "_").replace("-", "_")
            dep_result = context.task_results.get(dep_id)
            if dep_result and dep_result.state == TaskState.DONE:
                params[safe_key] = dep_result.output

        # Running context (if used)
        if "{running_context}" in template:
            params.setdefault(
                "running_context",
                context.running_context_text or "(still empty)",
            )

        # Fill missing keys with an empty string so that format() does not crash
        import string
        formatter = string.Formatter()
        used_keys = {
            key for _, key, _, _ in formatter.parse(template) if key
        }
        for key in used_keys:
            if key not in params:
                params[key] = f"(no value for {key})"

        return template.format(**params)

    def _try_parse_json(self, text: str) -> dict:
        """Try to parse JSON from the output, otherwise an empty dict."""
        if not text:
            return {}
        # JSON in ```json ... ``` Block
        text = text.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif text.startswith("```"):
            start = text.find("\n") + 1
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()

        if not (text.startswith("{") or text.startswith("[")):
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _dependencies_failed(
        self,
        task: SubTask,
        context: HarvestContext,
    ) -> bool:
        """Check whether at least one dependency is FAILED or SKIPPED."""
        for dep_id in task.depends_on:
            dep_result = context.task_results.get(dep_id)
            if not dep_result:
                continue
            if dep_result.state in (TaskState.FAILED, TaskState.SKIPPED):
                return True
        return False

    def _mark_skipped(self, task: SubTask, context: HarvestContext) -> None:
        """Mark a task as SKIPPED."""
        result = context.task_results[task.id]
        result.state = TaskState.SKIPPED
        result.error = "skipped because a dependency failed"
        result.started_at = datetime.now().isoformat()
        result.finished_at = result.started_at
        logger.info(f"Task {task.id}: SKIPPED ({result.error})")

    async def _notify_progress(self, event: str, data: dict) -> None:
        """Send a progress event (if a callback is set)."""
        if self.progress_callback:
            try:
                await self.progress_callback(event, data)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")


class RateLimitException(Exception):
    """Raised by the LLMCallable when the server returns 429.

    The DAG executor reacts with back-off and retry.
    """
    pass
