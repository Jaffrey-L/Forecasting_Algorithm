import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.api.linux_ops import compute_next_run_at
from src.api.platform_store import PlatformStore, utcnow_iso
from src.database.repositories import query_forecast_results, save_to_database
from src.forecasting.execution_bridge import get_data_from_db, process_single_spu


VALID_MODES = {"fast", "smart", "full"}
DEFAULT_SCOPE_MIN_WEEKS = 108
DEFAULT_SCOPE_RECENT_WEEKS = 4
DEFAULT_SCOPE_RECENT_NON_ZERO_WEEKS = 8
DEFAULT_SCOPE_RECENT_WINDOW_WEEKS = 12
DEFAULT_SCOPE_MAX_ZERO_RATIO_26W = 0.25
DEFAULT_SCOPE_MIN_RECENT4_TOTAL_SALES = 4.0
ACTIVE_RUN_STATUSES = {"queued", "running", "stopping"}
SCHEDULER_POLL_SECONDS = 30
SCHEDULER_RECOVERY_WAIT_SECONDS = 5

logger = logging.getLogger(__name__)


def normalize_spu_values(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip()
    return normalized[(normalized != "") & (normalized != "-")]


def _latest_active_week_buckets(weekly_sales: pd.DataFrame, count: int) -> List[pd.Timestamp]:
    weekly_totals = weekly_sales.groupby("week_bucket")["sales"].sum().sort_index()
    active_buckets = weekly_totals[weekly_totals > 0].index.tolist()
    return active_buckets[-count:]


class ForecastRuntimeManager:
    def __init__(self, store: PlatformStore, db_url: str):
        self.store = store
        self.db_url = db_url
        self._runs: Dict[str, threading.Event] = {}
        self._scheduler_stop = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_seen_keys: set[str] = set()
        self._schedule_run_lock = threading.Lock()

    def start_scheduler(self) -> None:
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        if self._scheduler_thread and not self._scheduler_thread.is_alive():
            logger.warning("Scheduler thread was not alive; restarting a fresh scheduler loop.")
        self._scheduler_stop.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Forecast scheduler loop started.")

    def stop_scheduler(self) -> None:
        self._scheduler_stop.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5)
            if self._scheduler_thread.is_alive():
                logger.warning("Scheduler loop did not stop within timeout.")

    def resolve_selection(self, selection_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        selection_type = (selection_type or "").lower()
        if selection_type != "all":
            raise ValueError("Only the standard all selection workflow is supported in the engineering workflow.")
        if selection_type == "all":
            all_spus, total_spus = self._load_all_spus_with_scope()
            return {
                "selection_type": "all",
                "selection_payload": {},
                "selected_spus": all_spus,
                "count": len(all_spus),
                "invalid_items": [],
                "preview": all_spus[:50],
                "scope_total_spus": total_spus,
                "scope_eligible_spus": len(all_spus),
                "scope_excluded_spus": max(total_spus - len(all_spus), 0),
            }
        raise ValueError(f"Unsupported selection type: {selection_type}")

    def create_run(
        self,
        mode: str,
        selection_type: str,
        selection_payload: Dict[str, Any],
        config_id: Optional[str] = None,
        trigger_source: str = "manual",
    ) -> Dict[str, Any]:
        if mode not in VALID_MODES:
            raise ValueError("mode must be one of fast, smart, full")
        run = self.store.create_run(
            config_id=config_id,
            trigger_source=trigger_source,
            mode=mode,
            selection_type=(selection_type or "").lower(),
            selection_payload=selection_payload,
            selected_spus=[],
        )
        stop_event = threading.Event()
        self._runs[run["id"]] = stop_event
        self.store.update_run(
            run["id"],
            current_spu="等待启动",
        )
        self._append_log(
            run["id"],
            "info",
            f"Run {run['id']} accepted and queued in {mode} mode.",
        )
        queued_run = self.store.get_run(run["id"])
        thread = threading.Thread(target=self._execute_run, args=(run["id"], stop_event), daemon=True)
        thread.start()
        return queued_run or self.store.get_run(run["id"])

    def stop_run(self, run_id: str) -> Dict[str, Any]:
        stop_event = self._runs.get(run_id)
        if stop_event:
            stop_event.set()
            self.store.update_run(run_id, status="stopping")
            self.store.add_log(run_id, "warning", "Stop requested by user.")
        run = self.store.get_run(run_id)
        if not run:
            raise ValueError("Run not found.")
        return run

    def run_from_config(self, config_id: str, trigger_source: str = "manual") -> Dict[str, Any]:
        config = self.store.get_config(config_id)
        if not config:
            raise ValueError("Config not found.")
        with self._schedule_run_lock:
            active_run = self._find_active_run_for_config(config_id)
            if active_run:
                logger.info(
                    "Skip run creation for config %s because active run %s is still %s.",
                    config_id,
                    active_run["id"],
                    active_run["status"],
                )
                return active_run
            return self.create_run(
                mode=config["mode"],
                selection_type=config["selection_type"],
                selection_payload=config["selection_payload"],
                config_id=config_id,
                trigger_source=trigger_source,
            )

    def get_results(self, **filters: Any) -> List[Dict[str, Any]]:
        return query_forecast_results(self.db_url, **filters)

    def _append_log(self, run_id: str, level: str, message: str) -> None:
        self.store.add_log(run_id, level, message)

    def _resolve_selection_with_heartbeat(
        self,
        run_id: str,
        selection_type: str,
        selection_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        error: Dict[str, Exception] = {}
        done = threading.Event()
        started = time.time()

        def _worker() -> None:
            try:
                result["resolved"] = self.resolve_selection(selection_type, selection_payload)
            except Exception as exc:  # pragma: no cover
                error["exc"] = exc
            finally:
                done.set()

        self.store.update_run(run_id, current_spu="Resolving selection scope")
        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        while not done.wait(timeout=15):
            elapsed = int(time.time() - started)
            self._append_log(run_id, "info", f"Selection scope resolving ({elapsed}s elapsed).")

        worker.join(timeout=1)
        if "exc" in error:
            raise error["exc"]
        return result.get("resolved", {})

    def _load_source_data_with_heartbeat(self, run_id: str) -> pd.DataFrame:
        result: Dict[str, Any] = {}
        error: Dict[str, Exception] = {}
        done = threading.Event()
        started = time.time()

        def _worker() -> None:
            try:
                result["df"] = get_data_from_db(self.db_url)
            except Exception as exc:  # pragma: no cover
                error["exc"] = exc
            finally:
                done.set()

        self._append_log(run_id, "info", "Starting source SQL query.")
        self.store.update_run(run_id, current_spu="SQL query running")
        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        while not done.wait(timeout=15):
            elapsed = int(time.time() - started)
            self._append_log(run_id, "info", f"SQL query still running ({elapsed}s elapsed).")

        worker.join(timeout=1)
        elapsed = time.time() - started
        if "exc" in error:
            raise error["exc"]
        df = result.get("df", pd.DataFrame())
        self._append_log(run_id, "info", f"SQL query finished in {elapsed:.1f}s, rows={len(df)}.")
        return df

    def _load_all_spus(self) -> List[str]:
        eligible_spus, _total_spus = self._load_all_spus_with_scope()
        return eligible_spus

    def _load_all_spus_with_scope(self) -> tuple[List[str], int]:
        df = get_data_from_db(self.db_url)
        if df.empty:
            return [], 0
        df.columns = [str(col).lower() for col in df.columns]
        if "date" not in df.columns or "spu" not in df.columns:
            all_spus = sorted(normalize_spu_values(df["spu"]).unique().tolist())
            return all_spus, len(all_spus)

        # The default "all" scope is restricted to SPUs with at least 108
        # weekly observations, and each of the latest 4 weekly buckets must
        # have non-zero sales so Linux scheduled runs skip recently inactive SPUs.
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["spu"] = normalize_spu_values(df["spu"])
        df["sales"] = pd.to_numeric(df.get("sales"), errors="coerce").fillna(0.0)
        df = df.dropna(subset=["spu", "date"])
        all_spus = sorted(df["spu"].unique().tolist())
        weekly_sales = (
            df.assign(week_bucket=df["date"].dt.to_period("W").dt.start_time)
            .groupby(["spu", "week_bucket"], as_index=False)["sales"]
            .sum()
        )
        weekly_counts = weekly_sales.groupby("spu")["week_bucket"].nunique()
        recent_week_buckets = _latest_active_week_buckets(weekly_sales, DEFAULT_SCOPE_RECENT_WEEKS)
        if len(recent_week_buckets) < DEFAULT_SCOPE_RECENT_WEEKS:
            return [], len(all_spus)
        recent_window_buckets = _latest_active_week_buckets(weekly_sales, DEFAULT_SCOPE_RECENT_WINDOW_WEEKS)
        recent_26_buckets = _latest_active_week_buckets(weekly_sales, 26)

        recent_sales = (
            weekly_sales[weekly_sales["week_bucket"].isin(recent_week_buckets)]
            .pivot(index="spu", columns="week_bucket", values="sales")
            .reindex(columns=recent_week_buckets, fill_value=0.0)
            .fillna(0.0)
        )
        active_recent_spus = recent_sales.index[(recent_sales > 0).all(axis=1)]
        recent_window_sales = (
            weekly_sales[weekly_sales["week_bucket"].isin(recent_window_buckets)]
            .pivot(index="spu", columns="week_bucket", values="sales")
            .reindex(columns=recent_window_buckets, fill_value=0.0)
            .fillna(0.0)
        )
        recent_non_zero_counts = (recent_window_sales > 0).sum(axis=1)
        recent_4_totals = recent_sales.sum(axis=1)
        recent_26_sales = (
            weekly_sales[weekly_sales["week_bucket"].isin(recent_26_buckets)]
            .pivot(index="spu", columns="week_bucket", values="sales")
            .reindex(columns=recent_26_buckets, fill_value=0.0)
            .fillna(0.0)
        )
        recent_26_zero_ratio = (recent_26_sales <= 0).sum(axis=1) / max(len(recent_26_buckets), 1)
        eligible_spus = active_recent_spus[
            (weekly_counts.reindex(active_recent_spus, fill_value=0) >= DEFAULT_SCOPE_MIN_WEEKS)
            & (recent_non_zero_counts.reindex(active_recent_spus, fill_value=0) >= DEFAULT_SCOPE_RECENT_NON_ZERO_WEEKS)
            & (recent_4_totals.reindex(active_recent_spus, fill_value=0.0) >= DEFAULT_SCOPE_MIN_RECENT4_TOTAL_SALES)
            & (recent_26_zero_ratio.reindex(active_recent_spus, fill_value=1.0) <= DEFAULT_SCOPE_MAX_ZERO_RATIO_26W)
        ].tolist()
        return sorted(eligible_spus), len(all_spus)

    def _execute_run(self, run_id: str, stop_event: threading.Event) -> None:
        run = self.store.get_run(run_id)
        if not run:
            return
        self.store.update_run(
            run_id,
            status="running",
            started_at=utcnow_iso(),
            current_spu="解析SPU范围",
            progress=0,
        )
        self._append_log(run_id, "info", f"Run {run_id} started in {run['mode']} mode.")
        try:
            self._append_log(
                run_id,
                "info",
                f"Resolving selection scope for {run['selection_type']} mode.",
            )
            resolved = self._resolve_selection_with_heartbeat(
                run_id,
                run["selection_type"],
                run.get("selection_payload", {}),
            )
            self.store.update_run(
                run_id,
                selected_spus_json=json.dumps(resolved["selected_spus"], ensure_ascii=False),
                total_count=resolved.get("count", len(resolved["selected_spus"])),
                processed_count=0,
                success_count=0,
                summary={
                    "scope_total_spus": resolved.get("scope_total_spus", len(resolved["selected_spus"])),
                    "scope_eligible_spus": resolved.get("scope_eligible_spus", len(resolved["selected_spus"])),
                    "scope_excluded_spus": resolved.get("scope_excluded_spus", 0),
                    "scope_min_weeks": DEFAULT_SCOPE_MIN_WEEKS,
                },
            )
            run["summary"] = {
                "scope_total_spus": resolved.get("scope_total_spus", len(resolved["selected_spus"])),
                "scope_eligible_spus": resolved.get("scope_eligible_spus", len(resolved["selected_spus"])),
                "scope_excluded_spus": resolved.get("scope_excluded_spus", 0),
                "scope_min_weeks": DEFAULT_SCOPE_MIN_WEEKS,
            }
            self._append_log(
                run_id,
                "info",
                f"Selection resolved: {resolved['count']} SPUs selected.",
            )
            self._append_log(run_id, "info", "Loading source dataset.")
            df_all = self._load_source_data_with_heartbeat(run_id)
            if df_all.empty:
                raise RuntimeError("No source data was returned from the database.")
            self._append_log(run_id, "info", "Normalizing source columns and data types.")
            self.store.update_run(run_id, current_spu="Preparing source data")
            df_all.columns = [str(col).lower() for col in df_all.columns]
            df_all["sales"] = pd.to_numeric(df_all["sales"], errors="coerce").fillna(0)
            df_all["date"] = pd.to_datetime(df_all["date"], format="mixed")
            df_all["spu"] = df_all["spu"].astype(str)
            df_all["sku"] = df_all["sku"].astype(str)
            selected_spus = resolved["selected_spus"] or sorted(df_all["spu"].unique().tolist())
            if selected_spus:
                df_all = df_all[df_all["spu"].isin(selected_spus)].copy()
            self._append_log(
                run_id,
                "info",
                f"Source scope filter applied: selected_spus={len(selected_spus)}, rows_after_filter={len(df_all)}.",
            )
            spus = sorted(df_all["spu"].unique().tolist())
            total_spus = len(spus)
            exog_cols = [col for col in df_all.columns if col in ["ad_cost", "price"]]
            self.store.update_run(run_id, total_count=total_spus)
            self._append_log(
                run_id,
                "info",
                f"Execution plan ready: total_spus={total_spus}, exog_cols={','.join(exog_cols) if exog_cols else 'none'}.",
            )
            successful = 0
            failures: List[Dict[str, str]] = []
            all_results: List[pd.DataFrame] = []
            policy_counts: Dict[str, int] = {"standard": 0, "conservative": 0, "unknown": 0}

            for index, spu in enumerate(spus, start=1):
                if stop_event.is_set():
                    self.store.update_run(run_id, status="stopped", finished_at=utcnow_iso())
                    self._append_log(run_id, "warning", "Run stopped before completion.")
                    return
                self.store.update_run(
                    run_id,
                    current_spu=spu,
                    processed_count=index - 1,
                    success_count=successful,
                    progress=int((index - 1) / max(total_spus, 1) * 100),
                )
                self.store.upsert_run_spu(run_id, spu, "running", message="Processing")
                self._append_log(run_id, "info", f"Processing SPU {spu} ({index}/{total_spus}).")
                self._append_log(run_id, "info", f"[SPU {spu}] 模型竞赛准备开始。")
                self._append_log(run_id, "info", f"[SPU {spu}] Model competition starting.")
                df_spu = df_all[df_all["spu"] == spu].copy()
                result_df, message, _viz, _profile = process_single_spu(
                    spu,
                    df_spu,
                    mode=run["mode"],
                    exog_cols=exog_cols,
                    collect_viz=False,
                    verbose=False,
                    log_fn=lambda line, current_spu=spu: self._append_log(
                        run_id, "info", f"[SPU {current_spu}] {line}"
                    ),
                )
                if result_df is not None:
                    all_results.append(result_df)
                    successful += 1
                    winner_algo = str(result_df["winner_algo"].iloc[0])
                    wmape = float(result_df["validation_wmape"].iloc[0])
                    if "policy=conservative" in (message or ""):
                        policy_counts["conservative"] += 1
                    elif "policy=standard" in (message or ""):
                        policy_counts["standard"] += 1
                    else:
                        policy_counts["unknown"] += 1
                    self.store.upsert_run_spu(
                        run_id,
                        spu,
                        "completed",
                        message=message,
                        winner_algo=winner_algo,
                        validation_wmape=wmape,
                    )
                else:
                    failures.append({"spu": spu, "reason": message})
                    self.store.upsert_run_spu(run_id, spu, "failed", message=message)
                self.store.update_run(
                    run_id,
                    processed_count=index,
                    success_count=successful,
                    progress=int(index / max(total_spus, 1) * 100),
                )

            summary: Dict[str, Any] = {
                "total_spus": total_spus,
                "successful_spus": successful,
                "failed_spus": len(failures),
                "failed_details": failures[:20],
                "policy_standard_spus": policy_counts["standard"],
                "policy_conservative_spus": policy_counts["conservative"],
                "policy_unknown_spus": policy_counts["unknown"],
                "scope_total_spus": run.get("summary", {}).get("scope_total_spus", total_spus),
                "scope_eligible_spus": run.get("summary", {}).get("scope_eligible_spus", total_spus),
                "scope_excluded_spus": run.get("summary", {}).get("scope_excluded_spus", 0),
                "scope_min_weeks": DEFAULT_SCOPE_MIN_WEEKS,
            }
            if all_results:
                final = pd.concat(all_results, ignore_index=True)
                save_to_database(final, self.db_url, run_id=run_id, config_id=run.get("config_id"))
                if "validation_wmape" in final.columns:
                    average_wmape = float(final["validation_wmape"].mean())
                    by_spu = final.groupby("spu")["validation_wmape"].mean().sort_values()
                    summary["average_wmape"] = average_wmape
                    summary["top_spus"] = by_spu.head(5).index.tolist()
                    summary["bottom_spus"] = by_spu.tail(5).index.tolist()
                self._append_log(run_id, "success", f"Saved {len(final)} forecast rows to the database.")
            self.store.update_run(
                run_id,
                status="completed",
                progress=100,
                processed_count=total_spus,
                success_count=successful,
                finished_at=utcnow_iso(),
                summary=summary,
            )
            self._append_log(run_id, "success", "Run completed.")
        except Exception as exc:
            self.store.update_run(
                run_id,
                status="failed",
                error_message=str(exc),
                finished_at=utcnow_iso(),
            )
            self._append_log(run_id, "error", f"Run failed: {exc}")
        finally:
            self._runs.pop(run_id, None)

    def _scheduler_loop(self) -> None:
        while not self._scheduler_stop.is_set():
            try:
                self._poll_schedules()
            except Exception:
                logger.exception("Scheduler poll failed unexpectedly; continuing after recovery wait.")
                self._scheduler_stop.wait(SCHEDULER_RECOVERY_WAIT_SECONDS)
                continue
            self._scheduler_stop.wait(SCHEDULER_POLL_SECONDS)

    def _poll_schedules(self) -> None:
        now_utc = datetime.now(timezone.utc)
        for schedule in self.store.list_schedules():
            if not schedule["enabled"]:
                continue
            due_slot = self._current_due_slot(schedule, now_utc)
            if due_slot is None:
                continue
            if self._already_triggered_for_slot(schedule, due_slot):
                continue
            dedupe_key = f"{schedule['id']}:{due_slot.strftime('%Y%m%d%H%M')}"
            if dedupe_key in self._scheduler_seen_keys:
                continue
            self._scheduler_seen_keys.add(dedupe_key)
            try:
                run = self.run_from_config(schedule["config_id"], trigger_source="schedule")
                self.store.mark_schedule_triggered(
                    schedule["id"],
                    due_slot.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                )
                if run.get("trigger_source") == "schedule":
                    logger.info(
                        "Schedule %s triggered config %s at slot %s, run_id=%s.",
                        schedule["id"],
                        schedule["config_id"],
                        due_slot.isoformat(),
                        run.get("id"),
                    )
                else:
                    logger.warning(
                        "Schedule %s skipped new run for config %s because active run_id=%s status=%s.",
                        schedule["id"],
                        schedule["config_id"],
                        run.get("id"),
                        run.get("status"),
                    )
            except Exception:
                self._scheduler_seen_keys.discard(dedupe_key)
                logger.exception(
                    "Failed to trigger schedule %s for config %s.",
                    schedule["id"],
                    schedule["config_id"],
                )

    def _find_active_run_for_config(self, config_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not config_id:
            return None
        for run in self.store.list_runs(limit=200):
            if run.get("config_id") != config_id:
                continue
            if run.get("status") in ACTIVE_RUN_STATUSES:
                return run
        return None

    @staticmethod
    def _resolve_timezone(timezone_name: Optional[str]) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name or "Asia/Shanghai")
        except Exception:
            return ZoneInfo("Asia/Shanghai")

    @staticmethod
    def _parse_iso_utc(raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        try:
            normalized = raw.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    def _already_triggered_for_slot(self, schedule: Dict[str, Any], due_slot_utc: datetime) -> bool:
        last_triggered_at = self._parse_iso_utc(schedule.get("last_triggered_at"))
        return bool(last_triggered_at and last_triggered_at >= due_slot_utc)

    def _current_due_slot(self, schedule: Dict[str, Any], now_utc: datetime) -> Optional[datetime]:
        timezone_name = str(schedule.get("timezone") or "Asia/Shanghai")
        tz = self._resolve_timezone(timezone_name)
        now_local = now_utc.astimezone(tz)
        current_slot = compute_next_run_at(
            weekday=int(schedule["weekday"]),
            hour=int(schedule["hour"]),
            minute=int(schedule["minute"]),
            timezone_name=timezone_name,
            now=now_local - timedelta(days=7),
        )
        current_slot_utc = current_slot.astimezone(timezone.utc)
        if current_slot_utc > now_utc:
            return None
        if now_utc - current_slot_utc >= timedelta(days=7):
            return None
        return current_slot_utc
