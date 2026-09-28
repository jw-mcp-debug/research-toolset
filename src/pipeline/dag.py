"""
DAG architecture of the research pipeline.

Basic infrastructure for splitting the pipeline into nodes. Deliberate
decisions:

1. **The pipeline is linear, not a general acyclic graph.**
   Despite "DAG" in the name, the research pipeline is sequential in
   practice, with an inner loop node for the search rounds. A real
   topological engine would be overkill here and would needlessly
   complicate stop-signal handling, progress reporting and use-case
   overrides. "DAG" refers to the modular node pipeline, not to a graph
   database.

2. **Nodes own the classifier/tool call, not the domain logic.**
   Separation of responsibilities: a node encapsulates one step and its
   I/O on the context, but the actual logic (`evaluate_coverage`,
   `judge_source_relevance` etc.) lives in the classifier modules. The
   node thus stays thin and testable.

3. **Nodes may mutate the context.**
   In a purely functional pipeline `ctx` would be immutable and every
   node would return a new `ctx`. That is not practical here, because
   HarvestContext is a dataclass with ~15 fields and copying/diffing it
   per node bloats the code. Instead, nodes explicitly mutate
   `ctx.<field>` and document it.

4. **The stop signal takes precedence over everything.**
   The stop signal is checked before every node run. A stopped signal
   leads to CancelledError, which the engine passes cleanly to the outer
   caller (the `run()` method sets ctx.status='cancelled').

5. **Errors are isolated.**
   An exception in node X does not necessarily stop the whole pipeline.
   `NodeResult.status='failed'` with `error` is an explicit status; the
   engine decides by the node's `fail_mode` whether the pipeline aborts
   or continues. The default is `continue` for classifier nodes (no
   single classifier should destroy the whole research) and `stop` for
   structural nodes (analysis, synthesis — without them there is nothing
   to deliver).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from src.core.stop_signal import StopSignal

logger = logging.getLogger(__name__)


# ── Status and result ───────────────────────────────────────────


class NodeStatus(str, Enum):
    """Possible results of a node run."""

    OK = "ok"                      # node succeeded
    SKIPPED = "skipped"            # applies_to → False or use-case override
    FAILED = "failed"              # exception inside the node
    STOP_PIPELINE = "stop_pipeline"  # the node signals: abort
                                   # (e.g. continue_decision = stop_done)


class FailMode(str, Enum):
    """What happens when the node returns `FAILED`?"""

    CONTINUE = "continue"  # pipeline continues (classifier nodes)
    STOP = "stop"          # pipeline aborts with ctx.status='error'


@dataclass
class NodeResult:
    """Result of a node run.

    Mutations of ctx happened outside this result — the result only says
    whether/how the node ran.
    """
    status: NodeStatus
    node_name: str = ""
    duration_seconds: float = 0.0
    error: Optional[BaseException] = None
    # Optional: structured metadata the node can produce (e.g. number of
    # filtered sources, classifier confidence). Used mainly for reporting
    # in the UI.
    metadata: dict = field(default_factory=dict)


# ── Node base class ────────────────────────────────────────────


class PipelineNode:
    """Base class of all pipeline nodes.

    Subclasses override:
      - `name` (class attribute, for logging/reporting)
      - `applies_to(ctx)` (default: always True)
      - `run(ctx)` (mandatory) — the actual work
      - `fail_mode` (class attribute, default CONTINUE)

    Responsibility contract of a node:

      * mutates `ctx` in clearly documented fields.
      * does NOT raise for foreseeable problems (LLM timeout, empty
        result); these are caught internally and reported as `OK` with a
        fallback marker in the classifier result.
      * raises ONLY for structural problems (broken configuration,
        missing mandatory fields in ctx).
      * long-running nodes check the stop signal periodically themselves
        (e.g. inside an asyncio.gather loop) and raise `CancelledError`.
    """

    name: str = "unnamed_node"
    fail_mode: FailMode = FailMode.CONTINUE

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        """Should this node run in this run?

        Default: always yes. Subclasses can check the use-case profile or
        mode and return a reason (for logging).

        Returns:
            (applies, reason) — with `applies=False`, `reason` says why
            the node was skipped.
        """
        return True, ""

    async def run(self, ctx: Any) -> Any:
        """The node's main work.

        Subclasses MUST override this. The return value is passed along in
        the `metadata` of the `NodeResult` — nodes do not need to return
        anything (None is fine).

        Convention: nodes may mutate `ctx`. Mutations are listed in the
        docstring of the concrete class.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() must be overridden",
        )


# ── Engine ────────────────────────────────────────────────────────


# Type alias for the progress callback. Nodes and engine pass
# (event_type, data) — both async.
ProgressCallback = Callable[[str, Any], Awaitable[None]]


class PipelineDAG:
    """Run a list of nodes in order.

    This is the execution core. A pipeline instance is:
      pipeline = PipelineDAG([
          QueryAnchorNode(llm),
          CoverageNode(llm),
          ...
      ])
      await pipeline.run(ctx, stop_signal, progress_callback)

    Behaviour:
      - nodes in `nodes` order.
      - before every node: stop-signal check.
      - applies_to False: the node is skipped and logged with a skipped
        result.
      - exception in the node: depending on `fail_mode` either continue
        or abort the pipeline.
      - the node returns STOP_PIPELINE: the engine stops, the remaining
        nodes are not run.

    Node results end up in `ctx.node_results` (a list), so that UI and
    diagnosis have access to the statistics (which nodes ran, which
    failed, how long they took).
    """

    def __init__(self, nodes: list[PipelineNode]):
        self.nodes = nodes

    async def run(
        self,
        ctx: Any,
        stop_signal: Optional["StopSignal"] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Any:
        """Run the nodes.

        Args:
            ctx: the shared context (typically HarvestContext). Every node
                 may mutate it.
            stop_signal: if set, checked before every node; raises
                         CancelledError when stopped.
            progress_callback: called with ('node_start', name) /
                               ('node_done', NodeResult) /
                               ('node_failed', NodeResult).

        Returns:
            The (mutated) ctx.
        """
        import asyncio
        import time

        # Collect node results if ctx supports it
        if not hasattr(ctx, "node_results"):
            try:
                ctx.node_results = []  # type: ignore
            except AttributeError:
                pass  # ctx is not mutable — then not

        for node in self.nodes:
            # Stop-signal check BEFORE the node — if the user stopped during a
            # node, the next check catches it. Nodes themselves are responsible
            # for internal stop checks.
            if stop_signal is not None:
                stop_signal.raise_if_stopped()

            # applies_to-Check
            applies, skip_reason = node.applies_to(ctx)
            if not applies:
                logger.debug(
                    "DAG: node %r skipped — %s",
                    node.name, skip_reason or "applies_to=False",
                )
                result = NodeResult(
                    status=NodeStatus.SKIPPED,
                    node_name=node.name,
                    metadata={"reason": skip_reason},
                )
                self._record(ctx, result)
                continue

            # Run the node
            t0 = time.perf_counter()
            if progress_callback is not None:
                try:
                    await progress_callback(
                        "node_start", {"name": node.name},
                    )
                except Exception as e:
                    logger.debug("Progress callback error ignored: %s", e)

            try:
                ret = await node.run(ctx)
                duration = time.perf_counter() - t0

                # STOP_PIPELINE? (the node signals a clean abort)
                if isinstance(ret, NodeResult):
                    result = ret
                    result.duration_seconds = duration
                    result.node_name = result.node_name or node.name
                else:
                    result = NodeResult(
                        status=NodeStatus.OK,
                        node_name=node.name,
                        duration_seconds=duration,
                        metadata=(ret if isinstance(ret, dict) else {}),
                    )

                self._record(ctx, result)
                if progress_callback is not None:
                    try:
                        await progress_callback("node_done", result)
                    except Exception as e:
                        logger.debug("Progress callback error ignored: %s", e)

                if result.status == NodeStatus.STOP_PIPELINE:
                    logger.info(
                        "DAG: node %r signals a pipeline stop",
                        node.name,
                    )
                    break

            except asyncio.CancelledError:
                # Stop signal or external cancel — pass it on cleanly
                logger.info("DAG: node %r cancelled", node.name)
                raise

            except Exception as e:
                duration = time.perf_counter() - t0
                logger.error(
                    "DAG: node %r failed (%s): %s",
                    node.name, node.fail_mode.value, e, exc_info=True,
                )
                result = NodeResult(
                    status=NodeStatus.FAILED,
                    node_name=node.name,
                    duration_seconds=duration,
                    error=e,
                )
                self._record(ctx, result)
                if progress_callback is not None:
                    try:
                        await progress_callback("node_failed", result)
                    except Exception:
                        pass

                if node.fail_mode == FailMode.STOP:
                    logger.error(
                        "DAG: node %r with fail_mode=STOP — "
                        "aborting the pipeline", node.name,
                    )
                    raise

                # fail_mode=CONTINUE: next node

        return ctx

    @staticmethod
    def _record(ctx: Any, result: NodeResult) -> None:
        """Append the result to ctx.node_results, if possible."""
        try:
            ctx.node_results.append(result)
        except (AttributeError, TypeError):
            pass  # ctx has no list — not critical
