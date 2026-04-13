#!/usr/bin/env python3
"""
Unified forecast execution entrypoint for manual and scheduled runs.

This script creates a persistent run record, delegates execution to the shared
runtime manager, and blocks until the run reaches a terminal state so Linux
schedulers can treat it as a normal job.
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.platform_store import PlatformStore
from src.api.runtime import ForecastRuntimeManager, VALID_MODES


DEFAULT_PLATFORM_DB = os.path.join(PROJECT_ROOT, "data", "platform_state.db")
DEFAULT_DB_URL = os.getenv(
    "SALES_FORECAST_DB_URL",
    "postgresql+psycopg2://postgres:vayiERty123@192.168.1.226:5432/finedatalink",
)
TERMINAL_STATUSES = {"completed", "failed", "stopped"}
ACTIVE_STATUSES = {"queued", "running", "stopping"}


@contextmanager
def process_lock(lock_path: str):
    lock_file = None
    released = False
    if not lock_path:
        yield
        return

    lock_dir = os.path.dirname(lock_path)
    try:
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        lock_file = open(lock_path, "a+", encoding="utf-8")
    except OSError as exc:
        # The service already has an outer flock guard; do not fail job startup
        # because an optional in-process lock path is not writable.
        print(f"WARN=lock disabled PATH={lock_path} REASON={exc}", flush=True)
        yield
        return
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except Exception:
        lock_file.close()
        raise RuntimeError(f"lock busy: {lock_path}")

    def _cleanup():
        nonlocal released
        if released:
            return
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            lock_file.close()
            released = True

    atexit.register(_cleanup)
    try:
        yield
    finally:
        _cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a forecast job through the shared status store.")
    parser.add_argument("--config-id", help="Existing config id to execute.")
    parser.add_argument("--mode", default="smart", choices=sorted(VALID_MODES))
    parser.add_argument("--selection-type", default="all", choices=["all", "manual", "sql"])
    parser.add_argument("--manual-spus", default="", help="Manual SPU list for selection_type=manual.")
    parser.add_argument("--sql-query", default="", help="SQL selection query for selection_type=sql.")
    parser.add_argument("--trigger-source", default="schedule", help="Recorded trigger source.")
    parser.add_argument("--platform-db", default=DEFAULT_PLATFORM_DB, help="Shared SQLite status db path.")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="Source database URL.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds.")
    parser.add_argument(
        "--active-run-timeout-hours",
        type=float,
        default=6.0,
        help="Treat active runs older than this timeout as stale and auto-close them.",
    )
    parser.add_argument(
        "--lock-file",
        default=os.path.join(PROJECT_ROOT, "data", "locks", "forecast-weekly.lock")
        if os.name == "posix"
        else "",
        help="Process lock file path to avoid concurrent scheduler jobs.",
    )
    return parser


def _parse_iso_utc(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def main() -> int:
    args = build_parser().parse_args()
    try:
        with process_lock(args.lock_file):
            store = PlatformStore(args.platform_db)
            manager = ForecastRuntimeManager(store=store, db_url=args.db_url)
            if hasattr(store, "get_current_run"):
                active_run = store.get_current_run()
            else:
                active_run = next(
                    (run for run in store.list_runs(limit=20) if run.get("status") in ACTIVE_STATUSES),
                    None,
                )
            if active_run and active_run.get("status") in ACTIVE_STATUSES:
                run_updated_at = _parse_iso_utc(active_run.get("updated_at") or active_run.get("started_at"))
                stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=args.active_run_timeout_hours)
                if run_updated_at is not None and run_updated_at < stale_cutoff:
                    store.update_run(
                        active_run["id"],
                        status="failed",
                        finished_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                        error_message=(
                            f"Auto-closed stale active run before new trigger; "
                            f"last_update={active_run.get('updated_at') or active_run.get('started_at')}"
                        ),
                    )
                    print(
                        f"STALE_RUN_CLOSED RUN_ID={active_run['id']} STATUS={active_run['status']} "
                        f"LAST_UPDATE={active_run.get('updated_at') or active_run.get('started_at')}",
                        flush=True,
                    )
                else:
                    print(f"SKIP=active run exists RUN_ID={active_run['id']} STATUS={active_run['status']}", flush=True)
                    return 0

            if args.config_id:
                run = manager.run_from_config(args.config_id, trigger_source=args.trigger_source)
            else:
                run = manager.create_run(
                    mode=args.mode,
                    selection_type=args.selection_type,
                    selection_payload={
                        "manual_spus": args.manual_spus,
                        "sql_query": args.sql_query,
                    },
                    trigger_source=args.trigger_source,
                )

            run_id = run["id"]
            print(f"RUN_ID={run_id}", flush=True)

            last_status = None
            while True:
                current = store.get_run(run_id)
                if not current:
                    print("ERROR=run disappeared from store", flush=True)
                    return 1
                if current["status"] != last_status:
                    print(
                        f"STATUS={current['status']} PROGRESS={current['progress']} "
                        f"PROCESSED={current['processed_count']} TOTAL={current['total_count']}",
                        flush=True,
                    )
                    last_status = current["status"]
                if current["status"] in TERMINAL_STATUSES:
                    if current["status"] == "completed":
                        return 0
                    print(f"ERROR={current.get('error_message') or current['status']}", flush=True)
                    return 1
                time.sleep(args.poll_interval)
    except RuntimeError as exc:
        print(f"SKIP={exc}", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
