from src.api.platform_store import PlatformStore
from src.api.runtime import ForecastRuntimeManager, parse_manual_spus


def test_parse_manual_spus_deduplicates_and_normalizes():
    parsed = parse_manual_spus("spu001, SPU001\nspu002 spu003")
    assert parsed["spus"] == ["SPU001", "SPU002", "SPU003"]
    assert parsed["invalid_items"] == []


def test_platform_store_config_roundtrip(tmp_path):
    store = PlatformStore(str(tmp_path / "platform_state.db"))
    config = store.create_config(
        name="Weekly Core SPUs",
        purpose="Ops review",
        mode="smart",
        selection_type="manual",
        selection_payload={"manual_spus": "SPU001\nSPU002"},
    )
    fetched = store.get_config(config["id"])
    assert fetched is not None
    assert fetched["name"] == "Weekly Core SPUs"
    assert fetched["selection_payload"]["manual_spus"] == "SPU001\nSPU002"


def test_fastapi_app_has_platform_routes():
    from src.api.app import app

    paths = {route.path for route in app.routes}
    assert "/api/forecast-jobs" in paths
    assert "/api/forecast-configs" in paths
    assert "/api/forecast-schedules" in paths
    assert "/api/linux-ops" in paths


def test_linux_ops_api_uses_snapshot_helper(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.app as app_module
    from src.api.platform_store import PlatformStore

    original_store = app_module.store
    monkeypatch.setattr(app_module, "store", PlatformStore(str(tmp_path / "platform_state.db")))

    def stub_snapshot(store, current_run=None, **kwargs):
        return {
            "captured_at": "2026-04-01T00:00:00Z",
            "current_run": None,
            "services": [],
            "next_schedule": None,
            "schedules": [],
            "logs": {"source": {"kind": "service", "label": "服务日志"}, "entries": []},
        }

    monkeypatch.setattr(app_module, "collect_linux_ops_snapshot", stub_snapshot)

    client = TestClient(app_module.app)
    response = client.get("/api/linux-ops")
    assert response.status_code == 200
    payload = response.json()
    assert payload["logs"]["source"]["label"] == "服务日志"
    monkeypatch.setattr(app_module, "store", original_store)


def test_frontend_entry_returns_new_console_markup():
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Overview" in response.text
    assert "mode-switch" in response.text
    assert 'data-mode="smart-only"' in response.text


def test_compatibility_analysis_status_idle_state(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.app as app_module
    from src.api.platform_store import PlatformStore

    original_store = app_module.store
    monkeypatch.setattr(app_module, "store", PlatformStore(str(tmp_path / "platform_state.db")))

    client = TestClient(app_module.app)
    response = client.get("/api/analysis-status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] is None
    assert payload["status"] == "idle"
    assert payload["progress"] == 0
    assert payload["processed_count"] == 0
    assert payload["total_count"] == 0
    assert payload["success_count"] == 0
    assert payload["current_spu"] is None
    assert payload["mode"] == "smart"
    assert payload["trigger_source"] is None
    assert payload["scope_min_weeks"] == 108
    assert payload["scope_total_spus"] == 0
    assert payload["scope_eligible_spus"] == 0
    assert payload["scope_excluded_spus"] == 0

    monkeypatch.setattr(app_module, "store", original_store)


def test_manual_selection_api_returns_preview():
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)
    response = client.post(
        "/api/spu-selection/resolve",
        json={"selection_type": "manual", "manual_spus": "spu001\nspu002,spu002"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["selected_spus"] == ["SPU001", "SPU002"]


def test_create_config_and_schedule_via_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.app as app_module
    from src.api.platform_store import PlatformStore

    original_store = app_module.store
    test_store = PlatformStore(str(tmp_path / "platform_state.db"))
    monkeypatch.setattr(app_module, "store", test_store)

    client = TestClient(app_module.app)
    config_resp = client.post(
        "/api/forecast-configs",
        json={
            "name": "Weekly Core SPUs",
            "purpose": "Ops review",
            "mode": "smart",
            "selection_type": "manual",
            "manual_spus": "SPU001\nSPU002",
        },
    )
    assert config_resp.status_code == 200
    config = config_resp.json()

    schedule_resp = client.post(
        "/api/forecast-schedules",
        json={
            "config_id": config["id"],
            "weekday": 0,
            "hour": 9,
            "minute": 0,
            "timezone": "Asia/Shanghai",
            "enabled": True,
        },
    )
    assert schedule_resp.status_code == 200
    assert schedule_resp.json()["config_id"] == config["id"]
    monkeypatch.setattr(app_module, "store", original_store)


def test_create_job_api_with_stubbed_manager(monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.app as app_module

    class StubManager:
        def create_run(self, mode, selection_type, selection_payload, config_id=None):
            return {
                "id": "run_test_001",
                "mode": mode,
                "selection_type": selection_type,
                "selection_payload": selection_payload,
                "selected_spus": ["SPU001", "SPU002"],
                "status": "queued",
                "progress": 0,
                "processed_count": 0,
                "total_count": 2,
                "success_count": 0,
                "current_spu": None,
                "summary": {},
            }

    monkeypatch.setattr(app_module, "manager", StubManager())
    client = TestClient(app_module.app)
    response = client.post(
        "/api/forecast-jobs",
        json={"mode": "smart", "selection_type": "manual", "manual_spus": "SPU001\nSPU002"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run_test_001"
    assert payload["run"]["total_count"] == 2


def test_compatibility_status_and_logs_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.app as app_module
    from src.api.platform_store import PlatformStore

    original_store = app_module.store
    test_store = PlatformStore(str(tmp_path / "platform_state.db"))
    monkeypatch.setattr(app_module, "store", test_store)

    run = test_store.create_run(
        config_id=None,
        trigger_source="manual",
        mode="smart",
        selection_type="manual",
        selection_payload={"manual_spus": "SPU001"},
        selected_spus=["SPU001"],
    )
    test_store.update_run(
        run["id"],
        status="running",
        progress=45,
        processed_count=3,
        total_count=8,
        success_count=2,
        current_spu="SPU001",
    )
    test_store.add_log(run["id"], "info", "Run started")

    client = TestClient(app_module.app)
    status_resp = client.get("/api/analysis-status")
    logs_resp = client.get("/api/analysis-logs")
    spus_resp = client.get("/api/completed-spus")

    assert status_resp.status_code == 200
    assert status_resp.json()["progress"] == 45
    assert status_resp.json()["trigger_source"] == "manual"
    assert logs_resp.status_code == 200
    assert logs_resp.json()["logs"][0]["message"] == "Run started"
    assert spus_resp.status_code == 200
    assert spus_resp.json()["completed_spus"] == []
    monkeypatch.setattr(app_module, "store", original_store)


def test_compatibility_start_analysis_defaults_to_all(monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.app as app_module

    class StubManager:
        def create_run(self, mode, selection_type, selection_payload, config_id=None, trigger_source="manual"):
            assert mode == "smart"
            assert selection_type == "all"
            assert selection_payload == {"manual_spus": "", "sql_query": ""}
            assert config_id is None
            assert trigger_source == "manual"
            return {
                "id": "run_legacy_001",
                "mode": mode,
                "status": "queued",
                "trigger_source": trigger_source,
            }

    monkeypatch.setattr(app_module, "manager", StubManager())
    client = TestClient(app_module.app)
    response = client.post("/api/start-analysis", json={})
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run_legacy_001"
    assert payload["mode"] == "smart"
    assert payload["trigger_source"] == "manual"


def test_stop_analysis_uses_latest_run(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.app as app_module
    from src.api.platform_store import PlatformStore

    original_store = app_module.store
    test_store = PlatformStore(str(tmp_path / "platform_state.db"))
    monkeypatch.setattr(app_module, "store", test_store)

    run = test_store.create_run(
        config_id=None,
        trigger_source="schedule",
        mode="smart",
        selection_type="all",
        selection_payload={},
        selected_spus=["SPU001"],
    )
    test_store.update_run(run["id"], status="running", total_count=1)

    class StubManager:
        def stop_run(self, run_id):
            assert run_id == run["id"]
            test_store.update_run(run_id, status="stopping")
            return test_store.get_run(run_id)

    monkeypatch.setattr(app_module, "manager", StubManager())

    client = TestClient(app_module.app)
    response = client.post("/api/stop-analysis")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run["id"]
    assert payload["status"] == "stopping"
    monkeypatch.setattr(app_module, "store", original_store)


def test_results_api_with_stubbed_manager(monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.app as app_module

    class StubManager:
        def get_results(self, **filters):
            assert filters["run_id"] == "run_123"
            assert filters["limit"] == 25
            return [
                {
                    "spu": "SPU001",
                    "run_date": "2026-03-20",
                    "forecast_target_date": "2026-03-27",
                    "spu_forecast_value": 123.4,
                    "winner_algo": "Prophet",
                    "validation_wmape": 0.12,
                    "run_id": "run_123",
                    "config_id": None,
                }
            ]

    monkeypatch.setattr(app_module, "manager", StubManager())
    client = TestClient(app_module.app)
    response = client.get("/api/forecast-results", params={"run_id": "run_123", "limit": 25})
    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["spu"] == "SPU001"
    assert payload[0]["winner_algo"] == "Prophet"


def test_project_management_dashboard_route():
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)
    response = client.get("/pm")
    assert response.status_code == 200
    assert "pm-dashboard.js" in response.text


def test_sql_selection_requires_single_spu_column(monkeypatch, tmp_path):
    import pandas as pd
    import src.api.runtime as runtime_module

    class StubEngine:
        def dispose(self):
            return None

    def stub_get_engine(_db_url):
        return StubEngine()

    def stub_read_sql(_query, con):
        assert con is not None
        return pd.DataFrame({"spu": ["spu001", "SPU002", "spu001"]})

    monkeypatch.setattr(runtime_module, "get_database_engine", stub_get_engine)
    monkeypatch.setattr(runtime_module.pd, "read_sql", stub_read_sql)

    store = PlatformStore(str(tmp_path / "platform_state.db"))
    manager = ForecastRuntimeManager(store=store, db_url="sqlite:///ignored.db")
    result = manager.resolve_selection("sql", {"sql_query": "select spu from some_table"})
    assert result["selected_spus"] == ["SPU001", "SPU002"]
    assert result["count"] == 2


def test_sql_selection_rejects_non_spu_shape(monkeypatch, tmp_path):
    import pandas as pd
    import src.api.runtime as runtime_module

    class StubEngine:
        def dispose(self):
            return None

    def stub_get_engine(_db_url):
        return StubEngine()

    def stub_read_sql(_query, con):
        assert con is not None
        return pd.DataFrame({"spu": ["SPU001"], "sku": ["SKU001"]})

    monkeypatch.setattr(runtime_module, "get_database_engine", stub_get_engine)
    monkeypatch.setattr(runtime_module.pd, "read_sql", stub_read_sql)

    store = PlatformStore(str(tmp_path / "platform_state.db"))
    manager = ForecastRuntimeManager(store=store, db_url="sqlite:///ignored.db")

    try:
        manager.resolve_selection("sql", {"sql_query": "select spu, sku from some_table"})
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "exactly one column named spu" in str(exc)


def test_manual_selection_is_not_filtered_by_all_scope_rules(tmp_path):
    store = PlatformStore(str(tmp_path / "platform_state.db"))
    manager = ForecastRuntimeManager(store=store, db_url="sqlite:///ignored.db")

    result = manager.resolve_selection("manual", {"manual_spus": "SPU001\nSPU002"})

    assert result["selected_spus"] == ["SPU001", "SPU002"]
    assert result["count"] == 2


def test_all_selection_excludes_spus_with_zero_sales_in_any_recent_week(monkeypatch, tmp_path):
    import pandas as pd
    import src.api.runtime as runtime_module

    weekly_dates = list(pd.date_range("2024-01-07", periods=10, freq="W"))
    df = pd.DataFrame(
        {
            "date": weekly_dates + weekly_dates + weekly_dates + weekly_dates[:6],
            "spu": ["SPU001"] * 10 + ["SPU002"] * 10 + ["SPU003"] * 10 + ["SPU004"] * 6,
            "sales": (
                [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                + [5, 6, 7, 8, 9, 10, 11, 12, 0, 14]
                + [5, 6, 7, 8, 9, 10, 0, 12, 13, 14]
                + [5, 6, 7, 8, 9, 10]
            ),
            "sku": ["SKU001"] * 10 + ["SKU002"] * 10 + ["SKU003"] * 10 + ["SKU004"] * 6,
        }
    )

    monkeypatch.setattr(runtime_module, "get_data_from_db", lambda _db_url: df)
    monkeypatch.setattr(runtime_module, "DEFAULT_SCOPE_MIN_WEEKS", 1)
    monkeypatch.setattr(runtime_module, "DEFAULT_SCOPE_RECENT_WEEKS", 4)

    store = PlatformStore(str(tmp_path / "platform_state.db"))
    manager = ForecastRuntimeManager(store=store, db_url="sqlite:///ignored.db")

    result = manager.resolve_selection("all", {})

    assert result["selected_spus"] == ["SPU001"]
    assert result["count"] == 1
    assert result["scope_total_spus"] == 4
    assert result["scope_eligible_spus"] == 1
    assert result["scope_excluded_spus"] == 3
