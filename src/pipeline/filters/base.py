"""
Filter architecture.

Filters share one protocol:
  - applies_to(ctx) -> bool : decides WHETHER the filter runs at all
  - apply(items, ctx) -> (kept, rejected, FilterStats)

So every filter has:
  1. its own activation condition (may check classifier output)
  2. visible statistics (for the UI)
  3. a clear distinction between "filter is not active" and "filter
     active, but rejected 0"

Filters are thus not switched on by mode constants or heuristic
thresholds in the orchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Statistics of one filter run (visible in logs and UI)."""
    name: str
    activated: bool = False
    activation_reason: str = ""
    total: int = 0
    kept: int = 0
    rejected: int = 0
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def loss_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.rejected / self.total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "activated": self.activated,
            "activation_reason": self.activation_reason,
            "total": self.total,
            "kept": self.kept,
            "rejected": self.rejected,
            "loss_rate": round(self.loss_rate, 3),
            "rejection_reasons_sample": self.rejection_reasons[:5],
        }


@runtime_checkable
class Filter(Protocol):
    """Protocol for extract filters."""

    name: str

    def applies_to(self, ctx: object) -> tuple[bool, str]:
        """Decide whether the filter becomes active.

        Returns:
            (active, reason) — reason is the explanation for UI/log.
        """
        ...

    async def apply(
        self,
        items: list,
        ctx: object,
    ) -> tuple[list, list, FilterStats]:
        """Apply the filter to the items.

        Returns:
            (kept, rejected, stats)
        """
        ...


async def run_filter_pipeline(
    items: list,
    ctx: object,
    filters: Iterable[Filter],
) -> tuple[list, dict[str, FilterStats]]:
    """Run several filters in sequence.

    Each filter decides itself whether it applies (`applies_to`).
    The order is the one configured in the use case.

    Returns:
        (kept_items, stats_per_filter) — kept_items are the items left
        after all filters; stats_per_filter is a dict
        filter name → FilterStats.
    """
    stats_map: dict[str, FilterStats] = {}
    current = list(items)

    for f in filters:
        active, reason = f.applies_to(ctx)
        stats = FilterStats(
            name=f.name,
            activated=active,
            activation_reason=reason,
            total=len(current),
        )
        if not active:
            stats.kept = len(current)
            stats_map[f.name] = stats
            logger.debug("Filter %s not active: %s", f.name, reason)
            continue

        kept, rejected, applied_stats = await f.apply(current, ctx)
        # Merge: the caller's activation reason + the filter's own detail statistics
        applied_stats.activated = True
        if not applied_stats.activation_reason:
            applied_stats.activation_reason = reason
        stats_map[f.name] = applied_stats
        current = kept

        if applied_stats.loss_rate > 0.5:
            logger.warning(
                "Filter %s: %d/%d discarded (%.1f%%)",
                f.name,
                applied_stats.rejected, applied_stats.total,
                applied_stats.loss_rate * 100,
            )

    return current, stats_map
