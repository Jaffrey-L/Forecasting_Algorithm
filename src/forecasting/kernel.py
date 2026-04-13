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
                else float("inf")
            )
            winner = {"name": "NaiveMean", "forecast": fallback_pred_test.tolist(), "wmape": fallback_wmape, "params": {}}
            all_results = [winner]
        else:
            all_results = valid_results
            winner = min(valid_results, key=lambda x: x["wmape"])

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

        return result_df, f"{winner['name']} (WMAPE: {winner['wmape']:.2%}) [{total_time:.1f}s]", viz_data, profile
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return None, f"閿欒: {str(exc)}", None, None


def save_to_database(*args, **kwargs):
    return _load_legacy_kernel().save_to_database(*args, **kwargs)


def main(*args, **kwargs):
    return _load_legacy_kernel().main(*args, **kwargs)
