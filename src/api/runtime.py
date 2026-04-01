import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from src.api.platform_store import PlatformStore, utcnow_iso
from src.database.repositories import get_database_engine, query_forecast_results, save_to_database


TOKEN_SPLIT_RE = re.compile(r"[\s,;|\r\n\t]+")
VALID_MODES = {"fast", "smart", "full"}
DEFAULT_SCOPE_MIN_WEEKS = 108


def normalize_spu_values(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip()
    return normalized[(normalized != "") & (normalized != "-")]


def parse_manual_spus(raw: str) -> Dict[str, Any]:
    tokens = [token.strip() for token in TOKEN_SPLIT_RE.split(raw or "") if token.strip()]
    seen = set()
    spus: List[str] = []
    invalid: List[str] = []
    for token in tokens:
        cleaned = token.upper()
        if len(cleaned) > 64:
            invalid.append(token)
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            spus.append(cleaned)
    return {"spus": spus, "invalid_items": invalid}


class ForecastRuntimeManager:
    def __init__(self, store: PlatformStore, db_url: str):
        self.store = store
        self.db_url = db_url
        self._runs: Dict[str, threading.Event] = {}
        self._scheduler_stop = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_seen_keys: set[str] = set()

    def start_scheduler(self) -> None:
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        self._scheduler_stop.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()

    def stop_scheduler(self) -> None:
        self._scheduler_stop.set()

    def resolve_selection(self, selection_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        selection_type = (selection_type or "").lower()
        if selection_type == "manual":
            parsed = parse_manual_spus(payload.get("manual_spus", ""))
            return {
                "selection_type": "manual",
                "selection_payload": {"manual_spus": payload.get("manual_spus", "")},
                "selected_spus": parsed["spus"],
                "count": len(parsed["spus"]),
                "invalid_items": parsed["invalid_items"],
                "preview": parsed["spus"][:50],
            }
        if selection_type == "sql":
            sql_query = (payload.get("sql_query") or "").strip()
            if not sql_query:
                raise ValueError("SQL selection requires sql_query.")
            if not sql_query.lower().startswith("select"):
                raise ValueError("Only SELECT statements are allowed for SQL selection.")
            engine = get_database_engine(self.db_url)
            try:
                df = pd.read_sql(text(sql_query), con=engine)
            finally:
                engine.dispose()
            if df.empty:
                values: List[str] = []
            else:
                columns = [str(column).lower() for column in df.columns]
                # MVP 阶段强约束：SQL 只能返回单列且列名必须是 spu
                if len(columns) != 1 or columns[0] != "spu":
                    raise ValueError("SQL must return exactly one column named spu.")
                series = df.iloc[:, 0]
                values = series.astype(str).tolist()
            parsed = parse_manual_spus("\n".join(values))
            return {
                "selection_type": "sql",
                "selection_payload": {"sql_query": sql_query},
                "selected_spus": parsed["spus"],
                "count": len(parsed["spus"]),
                "invalid_items": parsed["invalid_items"],
                "preview": parsed["spus"][:50],
            }
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
        resolved = self.resolve_selection(selection_type, selection_payload)
        run = self.store.create_run(
            config_id=config_id,
            trigger_source=trigger_source,
            mode=mode,
            selection_type=resolved["selection_type"],
            selection_payload=resolved["selection_payload"],
            selected_spus=resolved["selected_spus"],
        )
        self.store.update_run(
            run["id"],
            summary={
                "scope_total_spus": resolved.get("scope_total_spus", len(resolved["selected_spus"])),
                "scope_eligible_spus": resolved.get("scope_eligible_spus", len(resolved["selected_spus"])),
                "scope_excluded_spus": resolved.get("scope_excluded_spus", 0),
                "scope_min_weeks": DEFAULT_SCOPE_MIN_WEEKS,
            },
        )
        stop_event = threading.Event()
        self._runs[run["id"]] = stop_event
        thread = threading.Thread(target=self._execute_run, args=(run["id"], stop_event), daemon=True)
        thread.start()
        return self.store.get_run(run["id"])

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

    def _load_all_spus(self) -> List[str]:
        eligible_spus, _total_spus = self._load_all_spus_with_scope()
        return eligible_spus

    def _load_all_spus_with_scope(self) -> tuple[List[str], int]:
        from main import get_data_from_db

        df = get_data_from_db(self.db_url)
        if df.empty:
            return [], 0
        df.columns = [str(col).lower() for col in df.columns]
        if "date" not in df.columns or "spu" not in df.columns:
            all_spus = sorted(normalize_spu_values(df["spu"]).unique().tolist())
            return all_spus, len(all_spus)

        # The default "all" scope is restricted to SPUs with at least 108
        # weekly observations so Linux scheduled runs follow the same business rule.
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["spu"] = normalize_spu_values(df["spu"])
        df = df.dropna(subset=["spu"])
        all_spus = sorted(df["spu"].unique().tolist())
        weekly_counts = (
            df.dropna(subset=["date"])
            .assign(
                week_bucket=df["date"].dt.to_period("W").astype(str),
            )
            .groupby("spu")["week_bucket"]
            .nunique()
        )
        eligible_spus = weekly_counts[weekly_counts >= DEFAULT_SCOPE_MIN_WEEKS].index.tolist()
        return sorted(eligible_spus), len(all_spus)

    def _execute_run(self, run_id: str, stop_event: threading.Event) -> None:
        from main import get_data_from_db, process_single_spu

        run = self.store.get_run(run_id)
        if not run:
            return
        self.store.update_run(run_id, status="running", started_at=utcnow_iso())
        self._append_log(run_id, "info", f"Run {run_id} started in {run['mode']} mode.")
        try:
            df_all = get_data_from_db(self.db_url)
            if df_all.empty:
                raise RuntimeError("No source data was returned from the database.")
            df_all.columns = [str(col).lower() for col in df_all.columns]
            df_all["sales"] = pd.to_numeric(df_all["sales"], errors="coerce").fillna(0)
            df_all["date"] = pd.to_datetime(df_all["date"], format="mixed")
            df_all["spu"] = df_all["spu"].astype(str)
            df_all["sku"] = df_all["sku"].astype(str)
            selected_spus = run["selected_spus"] or sorted(df_all["spu"].unique().tolist())
            if selected_spus:
                df_all = df_all[df_all["spu"].isin(selected_spus)].copy()
            spus = sorted(df_all["spu"].unique().tolist())
            total_spus = len(spus)
            exog_cols = [col for col in df_all.columns if col in ["ad_cost", "price"]]
            self.store.update_run(run_id, total_count=total_spus)
            successful = 0
            failures: List[Dict[str, str]] = []
            all_results: List[pd.DataFrame] = []

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
                df_spu = df_all[df_all["spu"] == spu].copy()
                result_df, message, _viz, _profile = process_single_spu(
                    spu,
                    df_spu,
                    mode=run["mode"],
                    exog_cols=exog_cols,
                    collect_viz=False,
                    verbose=False,
                )
                if result_df is not None:
                    all_results.append(result_df)
                    successful += 1
                    winner_algo = str(result_df["winner_algo"].iloc[0])
                    wmape = float(result_df["validation_wmape"].iloc[0])
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
            self._poll_schedules()
            self._scheduler_stop.wait(30)

    def _poll_schedules(self) -> None:
        now = datetime.now()
        for schedule in self.store.list_schedules():
            if not schedule["enabled"]:
                continue
            if schedule["weekday"] != now.weekday():
                continue
            if schedule["hour"] != now.hour or schedule["minute"] != now.minute:
                continue
            dedupe_key = f"{schedule['id']}:{now.strftime('%Y%m%d%H%M')}"
            if dedupe_key in self._scheduler_seen_keys:
                continue
            self._scheduler_seen_keys.add(dedupe_key)
            self.store.mark_schedule_triggered(schedule["id"], utcnow_iso())
            try:
                self.run_from_config(schedule["config_id"], trigger_source="schedule")
            except Exception:
                pass
