import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text


def utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class PlatformStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS forecast_job_config (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                purpose TEXT,
                mode TEXT NOT NULL,
                selection_type TEXT NOT NULL,
                selection_payload_json TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS forecast_job_schedule (
                id TEXT PRIMARY KEY,
                config_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                weekday INTEGER NOT NULL,
                hour INTEGER NOT NULL,
                minute INTEGER NOT NULL,
                timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
                last_triggered_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS forecast_job_run (
                id TEXT PRIMARY KEY,
                config_id TEXT,
                trigger_source TEXT NOT NULL,
                mode TEXT NOT NULL,
                selection_type TEXT NOT NULL,
                selection_payload_json TEXT NOT NULL,
                selected_spus_json TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                processed_count INTEGER NOT NULL DEFAULT 0,
                total_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                current_spu TEXT,
                error_message TEXT,
                summary_json TEXT,
                started_at TEXT,
                finished_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS forecast_job_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS forecast_job_run_spu (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                spu TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                winner_algo TEXT,
                validation_wmape REAL,
                updated_at TEXT NOT NULL,
                UNIQUE(run_id, spu)
            )
            """,
        ]
        with self.engine.begin() as conn:
            for ddl in ddl_statements:
                conn.execute(text(ddl))

    @staticmethod
    def _json_load(raw: Optional[str], fallback: Any) -> Any:
        if not raw:
            return fallback
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return fallback

    @staticmethod
    def _hydrate_config(row: Dict[str, Any]) -> Dict[str, Any]:
        row["selection_payload"] = PlatformStore._json_load(
            row.pop("selection_payload_json", "{}"), {}
        )
        row["is_active"] = bool(row["is_active"])
        return row

    @staticmethod
    def _hydrate_schedule(row: Dict[str, Any]) -> Dict[str, Any]:
        row["enabled"] = bool(row["enabled"])
        return row

    @staticmethod
    def _hydrate_run(row: Dict[str, Any]) -> Dict[str, Any]:
        row["selection_payload"] = PlatformStore._json_load(
            row.pop("selection_payload_json", "{}"), {}
        )
        row["selected_spus"] = PlatformStore._json_load(
            row.pop("selected_spus_json", "[]"), []
        )
        row["summary"] = PlatformStore._json_load(row.pop("summary_json", "{}"), {})
        return row

    def create_config(
        self,
        name: str,
        purpose: str,
        mode: str,
        selection_type: str,
        selection_payload: Dict[str, Any],
        is_active: bool = True,
    ) -> Dict[str, Any]:
        config_id = uuid.uuid4().hex
        now = utcnow_iso()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO forecast_job_config
                    (id, name, purpose, mode, selection_type, selection_payload_json, is_active, created_at, updated_at)
                    VALUES
                    (:id, :name, :purpose, :mode, :selection_type, :selection_payload_json, :is_active, :created_at, :updated_at)
                    """
                ),
                {
                    "id": config_id,
                    "name": name,
                    "purpose": purpose,
                    "mode": mode,
                    "selection_type": selection_type,
                    "selection_payload_json": json.dumps(selection_payload, ensure_ascii=False),
                    "is_active": 1 if is_active else 0,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        return self.get_config(config_id)

    def list_configs(self) -> List[Dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT * FROM forecast_job_config ORDER BY updated_at DESC")
            ).mappings().all()
        return [self._hydrate_config(dict(row)) for row in rows]

    def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT * FROM forecast_job_config WHERE id = :id"),
                {"id": config_id},
            ).mappings().first()
        return self._hydrate_config(dict(row)) if row else None

    def create_or_update_schedule(
        self,
        config_id: str,
        weekday: int,
        hour: int,
        minute: int,
        timezone: str = "Asia/Shanghai",
        enabled: bool = True,
    ) -> Dict[str, Any]:
        now = utcnow_iso()
        with self.engine.begin() as conn:
            existing = conn.execute(
                text("SELECT id FROM forecast_job_schedule WHERE config_id = :config_id"),
                {"config_id": config_id},
            ).scalar_one_or_none()
            if existing:
                conn.execute(
                    text(
                        """
                        UPDATE forecast_job_schedule
                        SET weekday = :weekday,
                            hour = :hour,
                            minute = :minute,
                            timezone = :timezone,
                            enabled = :enabled,
                            updated_at = :updated_at
                        WHERE config_id = :config_id
                        """
                    ),
                    {
                        "config_id": config_id,
                        "weekday": weekday,
                        "hour": hour,
                        "minute": minute,
                        "timezone": timezone,
                        "enabled": 1 if enabled else 0,
                        "updated_at": now,
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO forecast_job_schedule
                        (id, config_id, enabled, weekday, hour, minute, timezone, created_at, updated_at)
                        VALUES
                        (:id, :config_id, :enabled, :weekday, :hour, :minute, :timezone, :created_at, :updated_at)
                        """
                    ),
                    {
                        "id": uuid.uuid4().hex,
                        "config_id": config_id,
                        "enabled": 1 if enabled else 0,
                        "weekday": weekday,
                        "hour": hour,
                        "minute": minute,
                        "timezone": timezone,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
        return self.get_schedule_by_config(config_id)

    def list_schedules(self) -> List[Dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT * FROM forecast_job_schedule ORDER BY updated_at DESC")
            ).mappings().all()
        return [self._hydrate_schedule(dict(row)) for row in rows]

    def get_schedule_by_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT * FROM forecast_job_schedule WHERE config_id = :config_id"),
                {"config_id": config_id},
            ).mappings().first()
        return self._hydrate_schedule(dict(row)) if row else None

    def mark_schedule_triggered(self, schedule_id: str, triggered_at: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE forecast_job_schedule
                    SET last_triggered_at = :last_triggered_at, updated_at = :updated_at
                    WHERE id = :id
                    """
                ),
                {
                    "id": schedule_id,
                    "last_triggered_at": triggered_at,
                    "updated_at": utcnow_iso(),
                },
            )

    def create_run(
        self,
        config_id: Optional[str],
        trigger_source: str,
        mode: str,
        selection_type: str,
        selection_payload: Dict[str, Any],
        selected_spus: List[str],
    ) -> Dict[str, Any]:
        run_id = uuid.uuid4().hex
        now = utcnow_iso()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO forecast_job_run
                    (id, config_id, trigger_source, mode, selection_type, selection_payload_json, selected_spus_json,
                     status, created_at, updated_at)
                    VALUES
                    (:id, :config_id, :trigger_source, :mode, :selection_type, :selection_payload_json, :selected_spus_json,
                     'queued', :created_at, :updated_at)
                    """
                ),
                {
                    "id": run_id,
                    "config_id": config_id,
                    "trigger_source": trigger_source,
                    "mode": mode,
                    "selection_type": selection_type,
                    "selection_payload_json": json.dumps(selection_payload, ensure_ascii=False),
                    "selected_spus_json": json.dumps(selected_spus, ensure_ascii=False),
                    "created_at": now,
                    "updated_at": now,
                },
            )
        return self.get_run(run_id)

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        payload = dict(fields)
        if "summary" in payload:
            payload["summary_json"] = json.dumps(payload.pop("summary"), ensure_ascii=False)
        payload["updated_at"] = utcnow_iso()
        assignments = ", ".join(f"{key} = :{key}" for key in payload.keys())
        payload["id"] = run_id
        with self.engine.begin() as conn:
            conn.execute(
                text(f"UPDATE forecast_job_run SET {assignments} WHERE id = :id"),
                payload,
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.begin() as conn:
            row = conn.execute(
                text("SELECT * FROM forecast_job_run WHERE id = :id"),
                {"id": run_id},
            ).mappings().first()
        return self._hydrate_run(dict(row)) if row else None

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT * FROM forecast_job_run ORDER BY created_at DESC LIMIT :limit"),
                {"limit": limit},
            ).mappings().all()
        return [self._hydrate_run(dict(row)) for row in rows]

    def add_log(self, run_id: str, level: str, message: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO forecast_job_log (run_id, level, message, created_at)
                    VALUES (:run_id, :level, :message, :created_at)
                    """
                ),
                {
                    "run_id": run_id,
                    "level": level,
                    "message": message,
                    "created_at": utcnow_iso(),
                },
            )

    def get_logs(self, run_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT run_id, level, message, created_at
                    FROM forecast_job_log
                    WHERE run_id = :run_id
                    ORDER BY id DESC
                    LIMIT :limit
                    """
                ),
                {"run_id": run_id, "limit": limit},
            ).mappings().all()
        return list(reversed([dict(row) for row in rows]))

    def upsert_run_spu(
        self,
        run_id: str,
        spu: str,
        status: str,
        message: Optional[str] = None,
        winner_algo: Optional[str] = None,
        validation_wmape: Optional[float] = None,
    ) -> None:
        with self.engine.begin() as conn:
            existing = conn.execute(
                text(
                    "SELECT id FROM forecast_job_run_spu WHERE run_id = :run_id AND spu = :spu"
                ),
                {"run_id": run_id, "spu": spu},
            ).scalar_one_or_none()
            payload = {
                "run_id": run_id,
                "spu": spu,
                "status": status,
                "message": message,
                "winner_algo": winner_algo,
                "validation_wmape": validation_wmape,
                "updated_at": utcnow_iso(),
            }
            if existing:
                payload["id"] = existing
                conn.execute(
                    text(
                        """
                        UPDATE forecast_job_run_spu
                        SET status = :status,
                            message = :message,
                            winner_algo = :winner_algo,
                            validation_wmape = :validation_wmape,
                            updated_at = :updated_at
                        WHERE id = :id
                        """
                    ),
                    payload,
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO forecast_job_run_spu
                        (run_id, spu, status, message, winner_algo, validation_wmape, updated_at)
                        VALUES
                        (:run_id, :spu, :status, :message, :winner_algo, :validation_wmape, :updated_at)
                        """
                    ),
                    payload,
                )

    def get_run_spus(self, run_id: str) -> List[Dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT spu, status, message, winner_algo, validation_wmape, updated_at
                    FROM forecast_job_run_spu
                    WHERE run_id = :run_id
                    ORDER BY updated_at DESC, spu ASC
                    """
                ),
                {"run_id": run_id},
            ).mappings().all()
        return [dict(row) for row in rows]
