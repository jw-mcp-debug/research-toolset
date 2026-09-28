"""Age-based deletion of stored research runs.

Every finished run is written to `<DATA_DIR>/<id>_<slug>/` (see
`HarvestContext.save`). Nothing in the application reads these
directories back, so they are kept only for the operator. This module
enforces `CLEANUP_MAX_AGE_DAYS`: directories whose last modification is
older than the limit are removed.

Gradio's own upload and download files are NOT handled here; they live
in GRADIO_TEMP_DIR and are cleaned up by Gradio (`delete_cache`).

Safety: only direct children of DATA_DIR whose name starts with the run
id format (YYYYMMDD_HHMMSS) are considered, so a misconfigured DATA_DIR
cannot cause unrelated files to be deleted.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}(_|$)")


def purge_expired_runs(data_dir: str | Path, max_age_days: int,
                       now: float | None = None) -> int:
    """Delete run directories older than `max_age_days`.

    `max_age_days <= 0` disables deletion. Returns the number of
    directories removed.
    """
    if max_age_days <= 0:
        return 0
    root = Path(data_dir)
    if not root.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - max_age_days * 86400
    removed = 0
    for entry in root.iterdir():
        if not entry.is_dir() or entry.is_symlink():
            continue
        if not _RUN_DIR_RE.match(entry.name):
            continue
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(entry)
            removed += 1
        except OSError as e:
            logger.warning("Could not delete expired run %s: %s", entry, e)
    if removed:
        logger.info("Retention: deleted %d run(s) older than %d day(s) from %s",
                    removed, max_age_days, root)
    return removed


def start_retention_worker(data_dir: str | Path, max_age_days: int,
                           interval_seconds: int = 6 * 3600) -> threading.Thread | None:
    """Run `purge_expired_runs` now and then every `interval_seconds`.

    Runs in a daemon thread so it never blocks shutdown. Returns None
    when retention is disabled.
    """
    if max_age_days <= 0:
        logger.info("Retention disabled (CLEANUP_MAX_AGE_DAYS=%s)", max_age_days)
        return None

    def _loop() -> None:
        while True:
            try:
                purge_expired_runs(data_dir, max_age_days)
            except Exception:  # never let the worker die silently
                logger.exception("Retention run failed")
            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, name="retention", daemon=True)
    t.start()
    return t
