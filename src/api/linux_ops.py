from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from src.api.platform_store import PlatformStore, utcnow_iso


ACTIVE_RUN_STATUSES = {"queued", "running", "stopping"}

SYSTEMD_UNITS = [
    {
        "label": "Forecast Dashboard V2 API",
        "unit": "forecast-dashboard-v2.service",
        "log_path": "/var/log/forecast-dashboard-v2.log",
    },
    {
        "label": "Weekly Forecast Job",
        "unit": "forecast-weekly.service",
        "log_path": "/var/log/forecast-weekly.log",
    },
    {
        "label": "Weekly Forecast Timer",
        "unit": "forecast-weekly.timer",
        "log_path": None,
    },
]

WEEKDAY_LABELS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _resolve_timezone(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name or "Asia/Shanghai")
    except Exception:
        return ZoneInfo("Asia/Shanghai")


def format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def compute_next_run_at(
    weekday: int,
    hour: int,
    minute: int,
    timezone_name: str = "Asia/Shanghai",
    now: Optional[datetime] = None,
) -> datetime:
    tz = _resolve_timezone(timezone_name)
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    else:
        current = current.astimezone(tz)
    candidate = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (weekday - candidate.weekday()) % 7
    if days_ahead == 0 and candidate <= current:
        days_ahead = 7
    return candidate + timedelta(days=days_ahead)


def tail_text_file(path: Optional[str], limit: int = 80) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = deque(fh, maxlen=limit)
    return [line.rstrip("\n") for line in lines]


def _run_systemctl(unit: str) -> dict[str, Any]:
    if platform.system() != "Linux" or shutil.which("systemctl") is None:
        return {
            "unit": unit,
            "available": False,
            "active_state": "unavailable",
            "sub_state": "unavailable",
            "unit_file_state": "unknown",
            "status_text": "systemctl unavailable on this host",
        }

    props = [
        "Description",
        "LoadState",
        "ActiveState",
        "SubState",
        "UnitFileState",
        "Result",
        "FragmentPath",
        "ExecMainPID",
        "MainPID",
        "ActiveEnterTimestamp",
        "InactiveEnterTimestamp",
        "NextElapseUSecRealtime",
        "LastTriggerUSecRealtime",
    ]
    command = ["systemctl", "show", unit]
    command.extend(f"--property={prop}" for prop in props)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception as exc:
        return {
            "unit": unit,
            "available": False,
            "active_state": "unknown",
            "sub_state": "unknown",
            "unit_file_state": "unknown",
            "status_text": f"systemctl query failed: {exc}",
        }

    if result.returncode != 0:
        return {
            "unit": unit,
            "available": False,
            "active_state": "unknown",
            "sub_state": "unknown",
            "unit_file_state": "unknown",
            "status_text": result.stderr.strip() or result.stdout.strip() or "systemctl query failed",
        }

    data: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value

    active_state = data.get("ActiveState", "unknown")
    sub_state = data.get("SubState", "unknown")
    unit_file_state = data.get("UnitFileState", "unknown")
    status_text = f"{active_state} / {sub_state}"
    if data.get("Result"):
        status_text = f"{status_text} ({data['Result']})"

    return {
        "unit": unit,
        "available": True,
        "description": data.get("Description"),
        "load_state": data.get("LoadState", "unknown"),
        "active_state": active_state,
        "sub_state": sub_state,
        "unit_file_state": unit_file_state,
        "result": data.get("Result"),
        "fragment_path": data.get("FragmentPath"),
        "exec_main_pid": data.get("ExecMainPID") or data.get("MainPID"),
        "active_enter_timestamp": data.get("ActiveEnterTimestamp"),
        "inactive_enter_timestamp": data.get("InactiveEnterTimestamp"),
        "next_elapse_realtime": data.get("NextElapseUSecRealtime"),
        "last_trigger_realtime": data.get("LastTriggerUSecRealtime"),
        "status_text": status_text,
    }


def build_schedule_overview(
    store: PlatformStore,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    configs = {config["id"]: config for config in store.list_configs()}
    schedules: list[dict[str, Any]] = []
    next_candidates: list[dict[str, Any]] = []

    for schedule in store.list_schedules():
        config = configs.get(schedule["config_id"], {})
        enabled = bool(schedule.get("enabled"))
        timezone_name = schedule.get("timezone", "Asia/Shanghai")
        next_run_at = None
        next_run_text = "已停用"
        if enabled:
            next_run_dt = compute_next_run_at(
                weekday=int(schedule["weekday"]),
                hour=int(schedule["hour"]),
                minute=int(schedule["minute"]),
                timezone_name=timezone_name,
                now=now,
            )
            next_run_at = next_run_dt.isoformat()
            next_run_text = format_dt(next_run_dt)
            next_candidates.append(
                {
                    "config_id": schedule["config_id"],
                    "config_name": config.get("name") or schedule["config_id"],
                    "next_run_at": next_run_at,
                    "next_run_text": next_run_text,
                    "sort_key": next_run_dt,
                }
            )

        schedules.append(
            {
                "schedule_id": schedule["id"],
                "config_id": schedule["config_id"],
                "config_name": config.get("name") or schedule["config_id"],
                "enabled": enabled,
                "weekday": int(schedule["weekday"]),
                "weekday_label": WEEKDAY_LABELS[int(schedule["weekday"])],
                "hour": int(schedule["hour"]),
                "minute": int(schedule["minute"]),
                "timezone": timezone_name,
                "last_triggered_at": schedule.get("last_triggered_at"),
                "next_run_at": next_run_at,
                "next_run_text": next_run_text,
            }
        )

    next_schedule = None
    if next_candidates:
        upcoming = min(next_candidates, key=lambda item: item["sort_key"])
        next_schedule = {
            key: value
            for key, value in upcoming.items()
            if key != "sort_key"
        }

    return {
        "schedules": schedules,
        "next_schedule": next_schedule,
    }


def _normalize_run_row(run: dict[str, Any]) -> dict[str, Any]:
    if not run:
        return {}
    return {
        "run_id": run.get("id"),
        "status": run.get("status"),
        "progress": run.get("progress"),
        "processed_count": run.get("processed_count"),
        "total_count": run.get("total_count"),
        "success_count": run.get("success_count"),
        "current_spu": run.get("current_spu"),
        "mode": run.get("mode"),
        "trigger_source": run.get("trigger_source"),
    }


def _normalize_log_rows(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "source": source,
                "timestamp": row.get("created_at"),
                "level": row.get("level", "info"),
                "message": row.get("message", ""),
            }
        )
    return normalized


def collect_linux_ops_snapshot(
    store: PlatformStore,
    current_run: Optional[dict[str, Any]] = None,
    now: Optional[datetime] = None,
    log_tail_limit: int = 120,
    systemctl_reader: Optional[Callable[[str], dict[str, Any]]] = None,
    file_tailer: Optional[Callable[[Optional[str], int], list[str]]] = None,
) -> dict[str, Any]:
    systemctl_reader = systemctl_reader or _run_systemctl
    file_tailer = file_tailer or tail_text_file
    schedule_overview = build_schedule_overview(store, now=now)
    active_run = current_run if current_run and current_run.get("status") in ACTIVE_RUN_STATUSES else None

    if active_run:
        log_rows = store.get_logs(active_run["id"], limit=log_tail_limit)
        log_entries = _normalize_log_rows(log_rows, source=f"run:{active_run['id']}")
        log_source = {
            "kind": "run",
            "label": "运行日志",
            "run_id": active_run["id"],
            "status": active_run.get("status"),
        }
    else:
        log_entries: list[dict[str, Any]] = []
        for unit in SYSTEMD_UNITS:
            if not unit["log_path"]:
                continue
            lines = file_tailer(unit["log_path"], limit=max(20, log_tail_limit // 2))
            if not lines:
                continue
            for line in lines:
                log_entries.append(
                    {
                        "source": unit["unit"],
                        "timestamp": None,
                        "level": "info",
                        "message": line,
                    }
                )
        if len(log_entries) > log_tail_limit:
            log_entries = log_entries[-log_tail_limit:]
        log_source = {
            "kind": "service",
            "label": "服务日志",
            "run_id": None,
            "status": "idle",
        }

    services = []
    for unit in SYSTEMD_UNITS:
        service_snapshot = systemctl_reader(unit["unit"])
        services.append(
            {
                **service_snapshot,
                "label": unit["label"],
                "log_path": unit["log_path"],
            }
        )

    return {
        "captured_at": utcnow_iso(),
        "current_run": _normalize_run_row(active_run) if active_run else None,
        "services": services,
        "next_schedule": schedule_overview["next_schedule"],
        "schedules": schedule_overview["schedules"],
        "logs": {
            "source": log_source,
            "entries": log_entries,
        },
    }
