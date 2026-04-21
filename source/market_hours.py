"""NYSE trading calendar and idempotency utilities for the daily job.

is_trading_day  — returns True if date is a NYSE business day
IdempotencyGuard — file-based marker that prevents duplicate runs
"""
import datetime as dt
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

try:
    import pandas_market_calendars as mcal
    _NYSE = mcal.get_calendar("NYSE")
    _HAS_MARKET_CAL = True
except ImportError:
    _HAS_MARKET_CAL = False
    log.warning(
        "pandas_market_calendars not installed; "
        "market-open check falls back to weekday-only (Mon–Fri)."
    )


def is_trading_day(date: dt.date = None) -> bool:
    """Return True if *date* is a NYSE trading day (not a holiday, not a weekend).

    Falls back to weekday check when pandas_market_calendars is absent.
    """
    if date is None:
        date = dt.date.today()
    if isinstance(date, dt.datetime):
        date = date.date()

    if _HAS_MARKET_CAL:
        schedule = _NYSE.schedule(
            start_date=date.isoformat(), end_date=date.isoformat()
        )
        return not schedule.empty

    return date.weekday() < 5  # Mon=0 … Fri=4


class IdempotencyGuard:
    """Prevents the daily job from running more than once per calendar day.

    Workflow:
        guard = IdempotencyGuard(Path("run_state"))
        if guard.already_ran():
            sys.exit(0)
        if not guard.acquire_lock():
            sys.exit(1)          # another process is running
        try:
            ... do work ...
            guard.mark_success()
        finally:
            guard.release_lock()

    Files created under *run_dir*:
        job.lock          — PID of the running process; deleted on release
        success_YYYY-MM-DD — empty marker; presence means today ran OK
    """

    def __init__(self, run_dir: Path):
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = self._run_dir / "job.lock"

    def _success_path(self, date: dt.date) -> Path:
        return self._run_dir / f"success_{date.isoformat()}"

    def already_ran(self, date: dt.date = None) -> bool:
        """Return True if today's success marker already exists."""
        if date is None:
            date = dt.date.today()
        return self._success_path(date).exists()

    def acquire_lock(self) -> bool:
        """Write a PID lock file.  Returns False if the lock is held by another process."""
        if self._lock_file.exists():
            existing_pid = self._lock_file.read_text().strip()
            log.warning("Lock file exists (PID %s). Another run may be active.", existing_pid)
            return False
        self._lock_file.write_text(str(os.getpid()))
        return True

    def release_lock(self):
        """Remove the PID lock file."""
        try:
            self._lock_file.unlink()
        except FileNotFoundError:
            pass

    def mark_success(self, date: dt.date = None):
        """Write the success marker for *date* (default: today)."""
        if date is None:
            date = dt.date.today()
        self._success_path(date).touch()
