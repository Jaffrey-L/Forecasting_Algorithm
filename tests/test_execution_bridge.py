def test_execution_bridge_reexports_kernel_api(monkeypatch):
    import src.forecasting.execution_bridge as bridge

    monkeypatch.setattr(bridge.forecast_kernel, "get_data_from_db", lambda db_url: f"bridge:{db_url}")
    monkeypatch.setattr(bridge.forecast_kernel, "process_single_spu", lambda *args, **kwargs: ("result", None, None, None))
    monkeypatch.setattr(bridge.forecast_kernel, "save_to_database", lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge.forecast_kernel, "main", lambda: "bridge-main")

    assert bridge.get_data_from_db("sqlite:///demo") == "bridge:sqlite:///demo"
    assert bridge.process_single_spu("SPU001", {"a": 1}) == ("result", None, None, None)
    assert bridge.save_to_database([1, 2, 3], "sqlite:///demo") is None
    assert bridge.main() == "bridge-main"


def test_runtime_uses_execution_bridge(monkeypatch, tmp_path):
    import pandas as pd
    import src.api.runtime as runtime_module
    from src.api.platform_store import PlatformStore

    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4, freq="W"),
            "spu": ["SPU001", "SPU001", "SPU002", "SPU002"],
            "sales": [10, 20, 30, 40],
            "sku": ["SKU001", "SKU001", "SKU002", "SKU002"],
        }
    )
    monkeypatch.setattr(runtime_module, "get_data_from_db", lambda _db_url: df)
    monkeypatch.setattr(runtime_module, "DEFAULT_SCOPE_MIN_WEEKS", 1)

    store = PlatformStore(str(tmp_path / "platform_state.db"))
    manager = runtime_module.ForecastRuntimeManager(store=store, db_url="sqlite:///demo")
    spus, total = manager._load_all_spus_with_scope()

    assert spus == ["SPU001", "SPU002"]
    assert total == 2
