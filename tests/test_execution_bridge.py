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
            "date": list(pd.date_range("2024-01-07", periods=8, freq="W")) * 2
            + list(pd.date_range("2024-01-07", periods=6, freq="W")),
            "spu": ["SPU001"] * 8 + ["SPU002"] * 8 + ["SPU003"] * 6,
            "sales": [10, 12, 14, 16, 18, 20, 22, 24] + [10, 12, 14, 16, 18, 0, 22, 24] + [8, 9, 10, 11, 12, 13],
            "sku": ["SKU001"] * 8 + ["SKU002"] * 8 + ["SKU003"] * 6,
        }
    )
    monkeypatch.setattr(runtime_module, "get_data_from_db", lambda _db_url: df)
    monkeypatch.setattr(runtime_module, "DEFAULT_SCOPE_MIN_WEEKS", 1)
    monkeypatch.setattr(runtime_module, "DEFAULT_SCOPE_RECENT_WEEKS", 4)

    store = PlatformStore(str(tmp_path / "platform_state.db"))
    manager = runtime_module.ForecastRuntimeManager(store=store, db_url="sqlite:///demo")
    spus, total = manager._load_all_spus_with_scope()

    assert spus == ["SPU001"]
    assert total == 3


def test_runtime_captures_model_competition_logs(monkeypatch, tmp_path):
    import pandas as pd
    import src.api.runtime as runtime_module
    from src.api.platform_store import PlatformStore

    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-07", periods=8, freq="W"),
            "spu": ["SPU001"] * 8,
            "sales": [10, 12, 14, 16, 18, 20, 22, 24],
            "sku": ["SKU001"] * 8,
        }
    )

    def stub_process_single_spu(spu, df_spu, **kwargs):
        print("运行 Prophet...")
        print("Prophet: WMAPE=12.34%")
        result_df = pd.DataFrame(
            {
                "spu": [spu],
                "winner_algo": ["Prophet"],
                "validation_wmape": [0.1234],
            }
        )
        return result_df, "Prophet (WMAPE: 12.34%) [1.0s]", None, None

    monkeypatch.setattr(runtime_module, "get_data_from_db", lambda _db_url: df)
    monkeypatch.setattr(runtime_module, "process_single_spu", stub_process_single_spu)
    monkeypatch.setattr(runtime_module, "save_to_database", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_module, "DEFAULT_SCOPE_MIN_WEEKS", 1)
    monkeypatch.setattr(runtime_module, "DEFAULT_SCOPE_RECENT_WEEKS", 4)

    store = PlatformStore(str(tmp_path / "platform_state.db"))
    manager = runtime_module.ForecastRuntimeManager(store=store, db_url="sqlite:///demo")
    run = store.create_run(
        config_id=None,
        trigger_source="manual",
        mode="smart",
        selection_type="manual",
        selection_payload={"manual_spus": "SPU001"},
        selected_spus=["SPU001"],
    )
    stop_event = runtime_module.threading.Event()
    manager._runs[run["id"]] = stop_event

    manager._execute_run(run["id"], stop_event)

    logs = store.get_logs(run["id"], limit=50)
    messages = [row["message"] for row in logs]
    assert any("[SPU SPU001] 运行 Prophet..." in message for message in messages)
    assert any("[SPU SPU001] Prophet: WMAPE=12.34%" in message for message in messages)
