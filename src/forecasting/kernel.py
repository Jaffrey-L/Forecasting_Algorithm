"""Canonical forecast execution kernel.

`process_single_spu` lives here now so the runtime can depend on a stable
`src.forecasting` surface instead of importing the legacy root `main.py`.
The legacy root module is still used for `save_to_database` and `main`.
"""

from __future__ import annotations

import datetime
import importlib
import json
import os
import time
from functools import lru_cache
from types import ModuleType

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.forecasting.models import SPUProfiler
from src.forecasting.sample_screening import screen_weekly_series
from src.forecasting.predictors import (
    calculate_wmape,
    clean_params_for_db,
    clean_series,
    extract_seasonal_factors_52week,
    get_current_week_end,
    plot_best_spu_style,
    predict_future,
    run_all_models,
    safe_predictions,
)
from src.forecasting.runtime_facade import (
    build_future_exog_frame,
    calculate_dynamic_shares,
    calculate_principal_dynamic_shares,
)


@lru_cache(maxsize=1)
def _load_legacy_kernel() -> ModuleType:
    return importlib.import_module("main")


def _build_training_data_query() -> str:
    spu_list_env = os.getenv("SPU_LIST", "").strip()
    spu_filter_sql = ""
    if spu_list_env:
        tokens = [s.strip() for s in spu_list_env.split(",") if s.strip()]
        if tokens:
            in_list = ",".join([f"'{t}'" for t in tokens])
            spu_filter_sql = f" and SPU in ({in_list})"

    return f"""
    with base as (
        select
        a."date" as report_date,
        local_sku,
        coalesce(a.principal_names, 'UNKNOWN') as principal_names,
        case
        when substring(local_sku,1,5)='RHNWB' then substring(local_sku,6,4)
        when substring(local_sku,1,2)='VY' then substring(local_sku,5,4)
        when substring(local_sku,1,2)='WB' then substring(local_sku,3,4)
        else '-' end as SPU,
        sum(afn_amount+mfn_amount+promotion_discount+refund_amount+cost_of_points_granted+inventory_credit+shared_fba_liquidation_proceeds+shared_fba_liquidation_proceeds_adjustments
        +shared_amazon_shipping_reimbursement+shared_safe_t_reimbursement+shared_netco_transaction+shared_reimbursements+shared_clawbacks+shared_commingling_vat_income+gift_wrap_credits
        +a_to_z_guarantee_claims+shared_others+shipping_cost) as 销售额,
        sum(a.volume) as 销量,
        avg(avg_net_amount) as 平均售价,
        sum(ads_sd_cost+ads_sp_cost+ads_sb_cost+ads_sbv_cost) as 广告费
        from lx_ods.查询订单利润_msku_cny_5年版 a
        left join lx_ods.查询订单利润_msku_cny_商品基础信息_5年版 b on a.__dm_key=b.__dm_key
        group by 1,2,3,4
    )
    select report_date as date, sum(销量) as sales, SPU as spu, local_sku as sku,
           max(principal_names) as principal_names,
           ROUND(SUM(-广告费)::NUMERIC, 2) as ad_cost, avg(平均售价) as price
    from base
    where 1=1{spu_filter_sql}
    group by report_date, SPU, local_sku
    order by report_date
    """


def get_data_from_db(db_url):
    t0 = time.time()
    query = _build_training_data_query()
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        print("尝试连接数据库...")
        with engine.connect() as conn:
            print("数据库连接成功！")
            print("开始执行SQL查询...")
            df = pd.read_sql(text(query), con=conn)
            print(f"SQL查询执行成功，获取到 {len(df)} 行数据")
        df.columns = [col.lower() for col in df.columns]
        print(f"数据获取完成！共加载 {len(df)} 行记录，耗时: {time.time() - t0:.1f} 秒")
        return df
    except Exception as exc:
        import traceback

        print(f"数据获取失败: {exc}")
        traceback.print_exc()
        return pd.DataFrame()
    finally:
        engine.dispose()
        print("数据库连接已关闭")


def _conservative_model_score(model: dict, test_series: pd.Series) -> float:
    preds = np.asarray(model.get("preds", []), dtype=float).flatten()
    actual = np.asarray(test_series.values, dtype=float).flatten()
    n = min(len(preds), len(actual))
    if n == 0:
        return float("inf")
    preds = np.maximum(preds[:n], 0.0)
    actual = actual[:n]

    wmape = float(model.get("wmape", float("inf")))
    if not np.isfinite(wmape):
        wmape = 9.999

    actual_zero_ratio = float(np.mean(actual == 0.0))
    pred_zero_ratio = float(np.mean(preds <= 1e-6))
    zero_mismatch = abs(pred_zero_ratio - actual_zero_ratio)

    pred_mean = float(np.mean(preds)) if n > 0 else 0.0
    actual_mean = float(np.mean(actual)) if n > 0 else 0.0
    over_bias = max(pred_mean - actual_mean, 0.0) / max(actual_mean, 1.0)

    pred_diff = np.diff(preds) if n > 1 else np.array([0.0])
    pred_jump = float(np.mean(np.abs(pred_diff))) / max(actual_mean, 1.0)

    # Composite score: accuracy first, then consistency and stability.
    return wmape + 0.35 * zero_mismatch + 0.15 * over_bias + 0.10 * pred_jump


def _standard_model_score(model: dict, test_series: pd.Series) -> float:
    """Lightweight tie-breaker for standard policy.

    Keep WMAPE as dominant signal while penalizing over-forecast and unstable
    trajectories when top candidates have similar validation errors.
    """
    preds = np.asarray(model.get("preds", []), dtype=float).flatten()
    actual = np.asarray(test_series.values, dtype=float).flatten()
    n = min(len(preds), len(actual))
    if n == 0:
        return float("inf")
    preds = np.maximum(preds[:n], 0.0)
    actual = actual[:n]

    wmape = float(model.get("wmape", float("inf")))
    if not np.isfinite(wmape):
        wmape = 9.999

    pred_mean = float(np.mean(preds)) if n > 0 else 0.0
    actual_mean = float(np.mean(actual)) if n > 0 else 0.0
    over_bias = max(pred_mean - actual_mean, 0.0) / max(actual_mean, 1.0)

    pred_diff = np.diff(preds) if n > 1 else np.array([0.0])
    pred_jump = float(np.mean(np.abs(pred_diff))) / max(actual_mean, 1.0)
    pred_zero_ratio = float(np.mean(preds <= 1e-6))
    actual_zero_ratio = float(np.mean(actual == 0.0))
    zero_mismatch = abs(pred_zero_ratio - actual_zero_ratio)

    # Lighter than conservative score, used only for close-call decisions.
    return wmape + 0.12 * over_bias + 0.08 * pred_jump + 0.06 * zero_mismatch


def _low_signal_model_score(model: dict, test_series: pd.Series, history_series: pd.Series) -> float:
    """Ranking score for sparse validation windows.

    In low-signal windows, many models hit the capped WMAPE (9.999), so we add
    penalties that favor conservative, non-explosive trajectories.
    """
    preds = np.asarray(model.get("preds", []), dtype=float).flatten()
    actual = np.asarray(test_series.values, dtype=float).flatten()
    n = min(len(preds), len(actual))
    if n == 0:
        return float("inf")
    preds = np.maximum(preds[:n], 0.0)
    actual = actual[:n]

    # Soft WMAPE uses relaxed non-zero requirement, useful for sparse windows.
    soft_wmape = float(calculate_wmape(actual, preds, min_non_zero_points=1))
    if not np.isfinite(soft_wmape):
        soft_wmape = 9.999

    hard_wmape = float(model.get("wmape", 9.999))
    if not np.isfinite(hard_wmape):
        hard_wmape = 9.999

    hist = pd.Series(history_series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    recent = hist.tail(min(12, len(hist))) if not hist.empty else pd.Series(dtype=float)
    recent_non_zero = recent[recent > 0]
    scale = float(recent_non_zero.median()) if not recent_non_zero.empty else float(recent.mean()) if not recent.empty else 1.0
    scale = max(scale, 1.0)

    mae_all = float(np.mean(np.abs(preds - actual)))
    zero_idx = actual <= 1e-6
    over_on_zero = float(np.mean(preds[zero_idx])) if np.any(zero_idx) else 0.0
    pred_zero_ratio = float(np.mean(preds <= 1e-6))
    actual_zero_ratio = float(np.mean(actual <= 1e-6))
    zero_mismatch = abs(pred_zero_ratio - actual_zero_ratio)

    # Extra penalty when model is in "hard-capped" error regime.
    cap_penalty = 0.35 if hard_wmape >= 9.0 else 0.0
    return (
        0.55 * soft_wmape
        + 0.20 * (mae_all / scale)
        + 0.15 * (over_on_zero / scale)
        + 0.10 * zero_mismatch
        + cap_penalty
    )


def _best_ensemble_candidate(valid_results: list[dict]) -> dict | None:
    ensemble_candidates = [
        candidate
        for candidate in valid_results
        if str(candidate.get("name", "")).startswith("Ensemble")
        and np.isfinite(float(candidate.get("wmape", float("inf"))))
    ]
    if not ensemble_candidates:
        return None
    return min(ensemble_candidates, key=lambda item: float(item.get("wmape", float("inf"))))


def _conservative_shortlist(valid_results: list[dict], wmape_margin: float = 0.015) -> list[dict]:
    finite_candidates = [
        candidate
        for candidate in valid_results
        if np.isfinite(float(candidate.get("wmape", float("inf"))))
    ]
    if not finite_candidates:
        return []
    best_wmape = min(float(candidate.get("wmape", float("inf"))) for candidate in finite_candidates)
    return [
        candidate
        for candidate in finite_candidates
        if float(candidate.get("wmape", float("inf"))) <= best_wmape + wmape_margin
    ]


def _apply_low_signal_post_rules(preds: np.ndarray, history_series: pd.Series, screening: dict) -> np.ndarray:
    adjusted = np.asarray(preds, dtype=float).flatten().copy()
    if adjusted.size == 0:
        return adjusted

    hist = pd.Series(history_series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if hist.empty:
        return np.maximum(adjusted, 0.0)

    recent = hist.tail(min(12, len(hist)))
    recent_non_zero = recent[recent > 0]
    anchor = float(recent_non_zero.median()) if not recent_non_zero.empty else float(recent.mean())
    anchor = max(anchor, 0.0)
    recent_std = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0
    recent_peak = float(recent.max()) if len(recent) > 0 else anchor

    cap = max(anchor * 2.2, recent_peak * 1.5, anchor + recent_std * 2.5, 1.0)
    adjusted = np.minimum(np.maximum(adjusted, 0.0), cap)

    max_step = max(anchor * 0.6, recent_std * 2.0, 1.0)
    for idx in range(1, len(adjusted)):
        lo = max(adjusted[idx - 1] - max_step, 0.0)
        hi = adjusted[idx - 1] + max_step
        adjusted[idx] = float(np.clip(adjusted[idx], lo, hi))

    if int(screening.get("recent_zero_weeks", 0)) >= 1 and anchor < 30:
        adjusted[0] = min(adjusted[0], anchor * 0.8)

    return np.maximum(adjusted, 0.0)


def process_single_spu(
    spu,
    df_spu,
    mode="smart",
    exog_cols=None,
    collect_viz=False,
    verbose=True,
    sku_accuracy_threshold=0.01,
    log_fn=None,
):
    t0 = time.time()
    try:
        profiler = SPUProfiler(verbose=verbose)
        df_spu_idx = df_spu.set_index("date").sort_index()

        series = df_spu_idx["sales"].resample("W").sum()
        current_week_end = get_current_week_end()
        series = series[series.index < current_week_end]
        original_series = series.copy()

        # Heuristic: if the latest weekly bucket is zero while recent history is active,
        # treat it as an incomplete ingestion week and exclude it from training/validation.
        if len(series) >= 20 and float(series.iloc[-1]) == 0.0:
            recent_active_weeks = int((series.iloc[-5:-1] > 0).sum()) if len(series) >= 5 else 0
            if recent_active_weeks >= 3:
                series = series.iloc[:-1]
                original_series = original_series.iloc[:-1]
                if log_fn is not None:
                    log_fn(f"SPU {spu} detected trailing zero week; trimmed latest week for stable validation.")

        has_exog, exog_series, used_exog = False, None, []
        if exog_cols:
            available = [c for c in exog_cols if c in df_spu_idx.columns]
            if available:
                exog_df = pd.DataFrame(index=series.index)
                if "ad_cost" in available:
                    exog_df["ad_cost"] = df_spu_idx["ad_cost"].resample("W").sum().fillna(0)
                if "price" in available:
                    df_spu_idx["revenue"] = df_spu_idx["sales"] * df_spu_idx["price"]
                    series_price = df_spu_idx["revenue"].resample("W").sum() / series.replace(0, np.nan)
                    exog_df["price"] = series_price.ffill().bfill().fillna(0)

                exog_series = exog_df[exog_df.index < current_week_end]
                has_exog, used_exog = True, available

        if len(series) > 156:
            series = series.iloc[-156:]
            original_series = original_series.iloc[-156:]
            if has_exog and exog_series is not None:
                exog_series = exog_series.iloc[-156:]
        if has_exog and exog_series is not None:
            exog_series = exog_series.reindex(series.index).ffill().bfill().fillna(0)

        screening = screen_weekly_series(series)
        if screening["insufficient_data"]:
            return None, f"数据不足 ({screening['history_weeks']}周)", None, None

        if screening["recommendation"] == "zero_override" and verbose:
            print(f"   Screening: {screening['recommendation']} | reasons={','.join(screening['reasons'])}")
        elif screening["recommendation"] == "conservative" and verbose:
            print(f"   Screening: {screening['recommendation']} | reasons={','.join(screening['reasons'])}")

        series_clean = clean_series(series)
        if len(series_clean) < 12:
            return None, "数据不足 (<12周)", None, None

        profile = profiler.analyze(spu, series_clean, original_series, exog_series)
        if verbose:
            profiler.print_profile(profile)

        test_len = min(16, max(8, len(series_clean) // 4))
        train, test = series_clean.iloc[:-test_len], series_clean.iloc[-test_len:]
        train_exog = exog_series.iloc[:-test_len] if has_exog else None
        test_exog = exog_series.iloc[-test_len:] if has_exog else None
        validation_non_zero_points = int((test > 0).sum())
        validation_total_sales = float(test.sum())
        low_signal_window = validation_non_zero_points < 6 or validation_total_sales < 120
        model_policy = (
            "conservative"
            if screening.get("recommendation") != "standard" or low_signal_window
            else "standard"
        )
        if log_fn is not None and model_policy == "conservative":
            log_fn(
                "SPU {} switched to conservative policy: recommendation={}, "
                "validation_non_zero_points={}, validation_total_sales={:.2f}.".format(
                    spu,
                    screening.get("recommendation"),
                    validation_non_zero_points,
                    validation_total_sales,
                )
            )
        future_dates = pd.date_range(series_clean.index[-1], periods=17, freq="W")[1:]

        if verbose:
            print(f"\nModel competition starting (mode={mode})...")
        if log_fn is not None:
            log_fn(f"SPU {spu} model competition starting, train={len(train)} weeks, test={len(test)} weeks.")
        all_results, base_results = run_all_models(
            train,
            test,
            mode,
            train_exog,
            test_exog,
            verbose=False,
            screening=screening,
            model_policy=model_policy,
            log_fn=(lambda message: log_fn(f"SPU {spu} | {message}")) if log_fn is not None else None,
        )
        valid_results = [result for result in all_results if np.isfinite(result.get("wmape", float("inf")))]
        if not valid_results:
            fallback_pred_test = np.full(len(test), float(test.mean()) if test.mean() > 0 else float(series_clean.mean()))
            fallback_wmape = float(
                np.sum(np.abs(test.values - fallback_pred_test)) / np.sum(np.abs(test.values))
                if (test > 0).sum() >= 4 and np.any(test.values != 0)
                else 9.999
            )
            winner = {"name": "NaiveMean", "forecast": fallback_pred_test.tolist(), "wmape": fallback_wmape, "params": {}}
            all_results = [winner]
        else:
            all_results = valid_results
            if model_policy == "conservative":
                wmape_best_candidate = min(valid_results, key=lambda item: float(item.get("wmape", float("inf"))))
                if low_signal_window:
                    ranked = sorted(
                        [
                            (
                                _low_signal_model_score(candidate, test, series_clean),
                                candidate,
                            )
                            for candidate in valid_results
                        ],
                        key=lambda item: item[0],
                    )
                    winner = ranked[0][1]
                    if log_fn is not None:
                        score_text = ", ".join([f"{item[1]['name']}={item[0]:.4f}" for item in ranked[:4]])
                        log_fn(f"SPU {spu} low-signal model ranking: {score_text}.")
                else:
                    shortlist = _conservative_shortlist(valid_results, wmape_margin=0.015)
                    candidates_for_rank = shortlist if shortlist else valid_results
                    ranked = sorted(
                        [
                            (
                                _conservative_model_score(candidate, test),
                                candidate,
                            )
                            for candidate in candidates_for_rank
                        ],
                        key=lambda item: item[0],
                    )
                    winner = ranked[0][1]
                    if log_fn is not None:
                        score_text = ", ".join([f"{item[1]['name']}={item[0]:.4f}" for item in ranked[:4]])
                        log_fn(f"SPU {spu} conservative model ranking: {score_text}.")

                winner_wmape = float(winner.get("wmape", float("inf")))
                wmape_best_value = float(wmape_best_candidate.get("wmape", float("inf")))
                if np.isfinite(winner_wmape) and np.isfinite(wmape_best_value):
                    guard_margin = 0.02 if low_signal_window else 0.015
                    if winner_wmape > wmape_best_value + guard_margin:
                        winner = wmape_best_candidate
                        if log_fn is not None:
                            log_fn(
                                f"SPU {spu} conservative wmape guardrail activated: "
                                f"use {winner['name']} ({wmape_best_value:.4f}) "
                                f"instead of higher-error candidate ({winner_wmape:.4f})."
                            )
                # Hybrid bridge: when signal is not extremely sparse, allow
                # stable ensemble to win if its error is close to conservative winner.
                ensemble_candidate = _best_ensemble_candidate(valid_results)
                if ensemble_candidate is not None:
                    winner_wmape = float(winner.get("wmape", float("inf")))
                    ensemble_wmape = float(ensemble_candidate.get("wmape", float("inf")))
                    hybrid_ready = (
                        validation_non_zero_points >= 8
                        and float(screening.get("zero_ratio", 1.0)) <= 0.45
                        and float(screening.get("recent_mean", 0.0)) >= 1.0
                    )
                    if (
                        hybrid_ready
                        and np.isfinite(winner_wmape)
                        and np.isfinite(ensemble_wmape)
                        and ensemble_wmape <= winner_wmape + 0.02
                    ):
                        winner = ensemble_candidate
                        if log_fn is not None:
                            log_fn(
                                f"SPU {spu} hybrid override enabled: "
                                f"choose {winner['name']} ({ensemble_wmape:.4f}) "
                                f"over conservative winner ({winner_wmape:.4f})."
                            )
            else:
                wmape_ranked = sorted(valid_results, key=lambda item: item.get("wmape", float("inf")))
                winner = wmape_ranked[0]
                if len(wmape_ranked) >= 2:
                    runner_up = wmape_ranked[1]
                    winner_wmape = float(winner.get("wmape", float("inf")))
                    runner_wmape = float(runner_up.get("wmape", float("inf")))
                    if np.isfinite(winner_wmape) and np.isfinite(runner_wmape):
                        wmape_gap = runner_wmape - winner_wmape
                        # If top models are close, use a stability-aware tie-breaker.
                        if wmape_gap <= 0.025:
                            close_candidates = [
                                candidate
                                for candidate in wmape_ranked
                                if float(candidate.get("wmape", 9.999)) - winner_wmape <= 0.025
                            ]
                            ranked = sorted(
                                [
                                    (
                                        _standard_model_score(candidate, test),
                                        candidate,
                                    )
                                    for candidate in close_candidates
                                ],
                                key=lambda item: item[0],
                            )
                            winner = ranked[0][1]
                            if log_fn is not None:
                                score_text = ", ".join(
                                    [
                                        f"{item[1]['name']}={item[0]:.4f}"
                                        for item in ranked[:4]
                                    ]
                                )
                                log_fn(f"SPU {spu} standard close-call ranking: {score_text}.")

        future_exog = None
        if has_exog and exog_series is not None:
            future_dates_exog = pd.date_range(series_clean.index[-1], periods=17, freq="W")[1:]
            future_exog = build_future_exog_frame(exog_series, future_dates_exog)

        final_preds = predict_future(series_clean, winner, 16, exog_series, future_exog, base_results)

        hist_cv = series_clean.std() / series_clean.mean() if series_clean.mean() > 0 else 0
        pred_cv = np.std(final_preds) / np.mean(final_preds) if np.mean(final_preds) > 0 else 0
        if verbose:
            print(f"   Volatility check: histCV={hist_cv:.3f}, predCV={pred_cv:.3f}, ratio={pred_cv / hist_cv:.2f}")
        if log_fn is not None:
            log_fn(f"SPU {spu} winner model: {winner['name']}, validation WMAPE {winner['wmape']:.2%}.")
            log_fn(f"SPU {spu} forecast generated for {len(future_dates)} future weeks.")
        if pred_cv < hist_cv * 0.3:
            if verbose:
                print("   Warning: forecast volatility is unusually low; fallback smoothing will be applied.")

        fallback_value = float(series_clean.iloc[-8:].mean())
        final_preds = safe_predictions(final_preds, fallback_value, winner["name"], history_series=series_clean)
        if model_policy == "conservative":
            final_preds = _apply_low_signal_post_rules(final_preds, series_clean, screening)

        total_time = time.time() - t0
        profile = profiler.update_with_results(profile, train, test, all_results, winner, final_preds, total_time)

        if verbose:
            profiler.print_model_competition(profile)
            profiler.print_forecast_summary(profile, future_dates, final_preds)

        share_json_list, share_df = calculate_dynamic_shares(df_spu_idx, spu, series_clean, future_dates)
        principal_share_json_list, _principal_share_df = calculate_principal_dynamic_shares(
            df_spu_idx, series_clean, future_dates
        )
        seasonal_factors_json = extract_seasonal_factors_52week(series_clean, period=52)

        sku_accuracy_json = None
        try:
            val_dates = test.index
            train_end_date = val_dates[0] - pd.Timedelta(days=1)
            df_train_idx = df_spu_idx[df_spu_idx.index <= train_end_date]
            train_spu_sales = series_clean[series_clean.index <= train_end_date]

            _, val_share_df = calculate_dynamic_shares(df_train_idx, spu, train_spu_sales, val_dates)
            winner_val_preds = pd.Series(winner["forecast"], index=val_dates)
            sku_val_preds = val_share_df.multiply(winner_val_preds, axis=0)

            sku_val_actual = (
                df_spu_idx[df_spu_idx.index.isin(val_dates)]
                .groupby([pd.Grouper(freq="W"), "sku"])["sales"]
                .sum()
                .unstack(fill_value=0)
            )
            sku_val_actual = sku_val_actual.reindex(index=val_dates, columns=sku_val_preds.columns, fill_value=0)

            sku_metrics = {}
            total_sku_sales = float(sku_val_actual.sum().sum())

            for sku in sku_val_preds.columns:
                actual = sku_val_actual[sku]
                pred = sku_val_preds[sku]
                sku_total = float(actual.sum())
                if total_sku_sales > 0 and sku_total / total_sku_sales < sku_accuracy_threshold:
                    continue

                actual_arr = np.asarray(actual)
                pred_arr = np.asarray(pred)
                mask = actual_arr != 0
                wmape = float(np.sum(np.abs(actual_arr[mask] - pred_arr[mask])) / np.sum(np.abs(actual_arr[mask]))) if np.any(mask) else 0.0
                sku_metrics[sku] = {
                    "wmape": round(wmape, 4),
                    "total_sales": sku_total,
                    "weight_in_spu": round(sku_total / total_sku_sales, 4) if total_sku_sales > 0 else 0,
                }

            sku_accuracy_json = json.dumps(sku_metrics, ensure_ascii=False)
        except Exception as exc:
            if verbose:
                print(f"   SKU 璇勪及澶辫触: {exc}")
            sku_accuracy_json = json.dumps({"error": str(exc)}, ensure_ascii=False)

        sku_future_df = share_df.multiply(final_preds, axis=0)
        safe_best_params = clean_params_for_db(winner.get("params"))

        result_df = pd.DataFrame(
            {
                "spu": spu,
                "run_date": datetime.date.today(),
                "forecast_target_date": [d.date() for d in future_dates],
                "spu_forecast_value": np.round(final_preds, 4),
                "sku_share_json": share_json_list,
                "principal_share_json": principal_share_json_list,
                "seasonal_factors_json": seasonal_factors_json,
                "sku_accuracy_json": sku_accuracy_json,
                "winner_algo": winner["name"],
                "validation_wmape": round(winner["wmape"], 4),
                "best_params": safe_best_params,
                "has_exog_features": has_exog,
                "exog_columns": ",".join(used_exog) if used_exog else None,
                "training_weeks": int(len(series_clean)),
                "data_end_date": series_clean.index[-1].date(),
            }
        )

        viz_data = (
            {
                "profile": profile,
                "train": train,
                "test": test,
                "results": all_results,
                "future": final_preds,
                "dates": future_dates,
                "winner": winner,
                "sku_future_df": sku_future_df,
            }
            if collect_viz
            else None
        )

        if collect_viz and verbose:
            plot_best_spu_style(
                profile,
                train,
                test,
                all_results,
                final_preds,
                future_dates,
                sku_future_df=sku_future_df,
                save_path=None,
                show_plot=True,
            )

        return (
            result_df,
            f"{winner['name']} (WMAPE: {winner['wmape']:.2%}, policy={model_policy}) [{total_time:.1f}s]",
            viz_data,
            profile,
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return None, f"閿欒: {str(exc)}", None, None


def save_to_database(*args, **kwargs):
    return _load_legacy_kernel().save_to_database(*args, **kwargs)


def main(*args, **kwargs):
    return _load_legacy_kernel().main(*args, **kwargs)
