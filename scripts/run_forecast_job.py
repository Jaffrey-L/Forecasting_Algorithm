#!/usr/bin/env python3
"""
Unified forecast execution entrypoint for manual and scheduled runs.

This script creates a persistent run record, delegates execution to the shared
runtime manager, and blocks until the run reaches a terminal state so Linux
schedulers can treat it as a normal job.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from src.api.platform_store import PlatformStore
from src.api.runtime import ForecastRuntimeManager, VALID_MODES


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PLATFORM_DB = os.path.join(PROJECT_ROOT, "data", "platform_state.db")
DEFAULT_DB_URL = os.getenv(
    "SALES_FORECAST_DB_URL",
    "postgresql+psycopg2://postgres:vayiERty123@192.168.1.226:5432/finedatalink",
)
TERMINAL_STATUSES = {"completed", "failed", "stopped"}


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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = PlatformStore(args.platform_db)
    manager = ForecastRuntimeManager(store=store, db_url=args.db_url)

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


if __name__ == "__main__":
    sys.exit(main())
