"""Age-based deletion of stored research runs (CLEANUP_MAX_AGE_DAYS)."""

import os
import time

from src.core.retention import purge_expired_runs

DAY = 86400


def _make(root, name, age_days, now):
    d = root / name
    d.mkdir()
    (d / "meta.json").write_text("{}")
    t = now - age_days * DAY
    os.utime(d, (t, t))
    return d


def test_expired_runs_are_deleted_fresh_ones_kept(tmp_path):
    now = time.time()
    old = _make(tmp_path, "20260101_120000_000000_abc123_query", 40, now)
    fresh = _make(tmp_path, "20260301_120000_000000_def456_query", 5, now)
    assert purge_expired_runs(tmp_path, 30, now=now) == 1
    assert not old.exists()
    assert fresh.exists()


def test_legacy_id_format_is_recognised(tmp_path):
    now = time.time()
    legacy = _make(tmp_path, "20260224_165631_some-query", 40, now)
    assert purge_expired_runs(tmp_path, 30, now=now) == 1
    assert not legacy.exists()


def test_foreign_entries_are_never_touched(tmp_path):
    now = time.time()
    foreign_dir = _make(tmp_path, "important-backup", 400, now)
    foreign_file = tmp_path / "20260101_120000_notes.txt"
    foreign_file.write_text("x")
    os.utime(foreign_file, (now - 400 * DAY, now - 400 * DAY))
    assert purge_expired_runs(tmp_path, 30, now=now) == 0
    assert foreign_dir.exists() and foreign_file.exists()


def test_zero_disables_deletion(tmp_path):
    now = time.time()
    old = _make(tmp_path, "20260101_120000_000000_abc123_query", 400, now)
    assert purge_expired_runs(tmp_path, 0, now=now) == 0
    assert old.exists()


def test_missing_data_dir_is_harmless(tmp_path):
    assert purge_expired_runs(tmp_path / "missing", 30) == 0


def test_symlinked_run_dir_is_not_followed(tmp_path):
    now = time.time()
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    data = tmp_path / "data"
    data.mkdir()
    link = data / "20260101_120000_000000_abc123_link"
    link.symlink_to(target)
    assert purge_expired_runs(data, 30, now=now + 400 * DAY) == 0
    assert (target / "keep.txt").exists()
