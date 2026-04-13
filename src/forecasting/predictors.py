import pandas as pd
import numpy as np
import pmdarima as pm
import json
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import itertools
import time
import random
from functools import lru_cache, partial
from src.forecasting.models import *

forecast_kernel = None
MAX_WMAPE_CAP = 9.999


def _series_to_float_series(series):
    return pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def _infer_model_regime(train, screening=None):
    clean = _series_to_float_series(train)
    history_weeks = len(clean)
    recent = clean.tail(min(12, history_weeks))
    recent_zero_weeks = int((recent == 0).sum()) if len(recent) > 0 else 0
    recent_mean = float(recent.mean()) if len(recent) > 0 else 0.0
    prior = clean.iloc[:-len(recent)] if history_weeks > len(recent) else pd.Series(dtype=float)
    prior_mean = float(prior.mean()) if len(prior) > 0 else 0.0
    zero_ratio = float((clean == 0).mean()) if history_weeks > 0 else 1.0

    if screening is not None:
        recent_zero_weeks = int(screening.get("recent_zero_weeks", recent_zero_weeks))
        recent_mean = float(screening.get("recent_mean", recent_mean))
        prior_mean = float(screening.get("prior_mean", prior_mean))
        zero_ratio = float(screening.get("zero_ratio", zero_ratio))

    return {
        "history_weeks": history_weeks,
        "recent_zero_weeks": recent_zero_weeks,
        "recent_mean": recent_mean,
        "prior_mean": prior_mean,
        "zero_ratio": zero_ratio,
        "seasonal_ready": history_weeks >= 52,
        "stable": history_weeks >= 26 and zero_ratio < 0.35,
        "zero_heavy": zero_ratio >= 0.5 or recent_zero_weeks >= 4,
    }


def _build_zero_aware_forecast(train, n_steps, screening=None):
    clean = _series_to_float_series(train)
    if clean.empty:
        return np.zeros(n_steps)

    regime = _infer_model_regime(clean, screening)
    recent = clean.tail(min(12, len(clean)))
    recent_non_zero = recent[recent > 0]
    anchor = float(recent_non_zero.median()) if not recent_non_zero.empty else float(recent.mean())
    anchor = max(anchor, 0.0)

    if regime["zero_heavy"] or (screening and screening.get("recommendation") == "zero_override"):
        if regime["recent_zero_weeks"] >= 4 or regime["recent_mean"] == 0:
            return np.zeros(n_steps)
        decay = np.linspace(1.0, 0.25, n_steps)
        return np.maximum(anchor * decay, 0.0)

    if len(clean) >= 52:
        season = clean.iloc[-52:].to_numpy()
        seasonal = np.resize(season, n_steps)
        blended = 0.65 * seasonal + 0.35 * anchor
        return np.maximum(blended, 0.0)

    return np.maximum(np.full(n_steps, anchor), 0.0)


def _get_forecast_kernel():
    if forecast_kernel is not None:
        return forecast_kernel
    from src.forecasting import kernel as loaded_kernel

    return loaded_kernel


@lru_cache(maxsize=1)
def _check_prophet_runtime():
    """Preflight Prophet runtime once to avoid repeated per-SPU hard failures."""
    try:
        test_df = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-07", periods=20, freq="W"),
                "y": np.linspace(10.0, 30.0, 20),
            }
        )
        model = Prophet(
            weekly_seasonality=False,
            yearly_seasonality=False,
            daily_seasonality=False,
        )
        model.fit(test_df)
        model.predict(test_df.tail(2))
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def clean_series(series):
    series = series.copy()
    series = series[series >= 0]
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    series = series[(series >= lower_bound) & (series <= upper_bound)]
    series = series.fillna(0)
    return series


def get_current_week_end():
    today = pd.Timestamp.today()
    return today - pd.Timedelta(days=today.weekday() + 1)


def calculate_wmape(y_true, y_pred, min_non_zero_points=1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    mask = y_true != 0
    if np.count_nonzero(mask) < min_non_zero_points:
        return MAX_WMAPE_CAP if min_non_zero_points > 1 else 0.0
    denominator = np.sum(np.abs(y_true[mask]))
    if denominator == 0:
        return MAX_WMAPE_CAP if min_non_zero_points > 1 else 0.0
    value = np.sum(np.abs(y_true[mask] - y_pred[mask])) / denominator
    if not np.isfinite(value):
        return MAX_WMAPE_CAP
    return float(min(value, MAX_WMAPE_CAP))


def run_prophet(train, test, train_exog=None, test_exog=None, verbose=False, screening=None):
    try:
        regime = _infer_model_regime(train, screening)
        df_train = pd.DataFrame({'ds': train.index, 'y': train.values})

        if df_train.empty:
            if verbose:
                print("Prophet 澶辫触: 璁粌鏁版嵁涓虹┖")
            else:
                print("Prophet 澶辫触: 璁粌鏁版嵁涓虹┖")
            return None

        if train_exog is not None:
            if not train_exog.empty:
                train_exog_aligned = train_exog.reindex(train.index)
                for col in train_exog_aligned.columns:
                    if not train_exog_aligned[col].isna().all():
                        df_train[col] = train_exog_aligned[col].values
        seasonality_mode = 'multiplicative' if regime["history_weeks"] >= 26 and regime["zero_ratio"] < 0.5 else 'additive'
        changepoint_prior_scale = 0.05 if regime["stable"] else 0.12
        seasonality_prior_scale = 5.0 if regime["history_weeks"] >= 104 else 10.0
        changepoint_range = 0.9 if regime["history_weeks"] >= 52 else 0.8
        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            changepoint_range=changepoint_range,
            weekly_seasonality=False,
            yearly_seasonality=False,
        )
        if regime["history_weeks"] >= 13:
            model.add_seasonality(name='13w', period=13, fourier_order=3)
        if regime["history_weeks"] >= 52:
            model.add_seasonality(name='52w', period=52, fourier_order=5)
        if train_exog is not None:
            if not train_exog.empty:
                for col in train_exog.columns:
                    model.add_regressor(col)
        model.fit(df_train)
        df_test = pd.DataFrame({'ds': test.index})
        if test_exog is not None:
            if not test_exog.empty:
                test_exog_aligned = test_exog.reindex(test.index)
                for col in test_exog_aligned.columns:
                    if not test_exog_aligned[col].isna().all():
                        df_test[col] = test_exog_aligned[col].values
        forecast = model.predict(df_test)
        y_pred = forecast['yhat'].values
        y_pred = np.maximum(y_pred, 0)
        if regime["zero_heavy"]:
            y_pred = np.minimum(y_pred, max(regime["recent_mean"] * 1.2, 1.0))
        wmape = calculate_wmape(test.values, y_pred, min_non_zero_points=4)
        return {'name': 'Prophet', 'wmape': wmape, 'preds': y_pred, 'model': model, 'params': model.params if hasattr(model, 'params') else {}}
    except Exception as e:
        if verbose:
            print(f"Prophet 澶辫触: {e}")
            import traceback
            traceback.print_exc()
        return None


def run_xgboost(train, test, train_exog=None, test_exog=None, verbose=False, screening=None):
    try:
        regime = _infer_model_regime(train, screening)
        fe = FeatureEngineer()
        X_train, y_train = fe.make_features(pd.DataFrame(train), train_exog)
        
        # 妫€鏌_train鎴杫_train鏄惁涓虹┖
        if X_train.empty or y_train.empty:
            if verbose:
                print("XGBoost 澶辫触: 璁粌鏁版嵁涓虹┖")
            else:
                print("XGBoost 澶辫触: 璁粌鏁版嵁涓虹┖")
            return None
            
        X_test, _ = fe.make_features_for_prediction(pd.DataFrame(test), test_exog)
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        if regime["zero_heavy"]:
            params.update({"max_depth": 3, "min_child_weight": 6, "learning_rate": 0.03})
        model = XGBRegressor(**params)
        eval_set = None
        fit_kwargs = {"verbose": False}
        if len(X_train) >= 20:
            split = max(4, int(len(X_train) * 0.2))
            if len(X_train) - split >= 12:
                X_fit, X_valid = X_train.iloc[:-split], X_train.iloc[-split:]
                y_fit, y_valid = y_train.iloc[:-split], y_train.iloc[-split:]
                eval_set = [(X_valid, y_valid)]
                fit_kwargs.update({"eval_set": eval_set, "early_stopping_rounds": 25})
                model.fit(X_fit, y_fit, **fit_kwargs)
            else:
                model.fit(X_train, y_train, **fit_kwargs)
        else:
            model.fit(X_train, y_train, **fit_kwargs)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)
        if regime["zero_heavy"]:
            y_pred = np.minimum(y_pred, max(regime["recent_mean"] * 1.5, 1.0))
        wmape = calculate_wmape(test.values, y_pred, min_non_zero_points=4)
        return {'name': 'XGBoost', 'wmape': wmape, 'preds': y_pred, 'model': model, 'params': model.get_params()}
    except Exception as e:
        if verbose:
            print(f"XGBoost 澶辫触: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"XGBoost 澶辫触: {e}")
        return None


def run_lightgbm(train, test, train_exog=None, test_exog=None, verbose=False, screening=None):
    try:
        regime = _infer_model_regime(train, screening)
        fe = FeatureEngineer()
        X_train, y_train = fe.make_features(pd.DataFrame(train), train_exog)
        
        # 妫€鏌_train鎴杫_train鏄惁涓虹┖
        if X_train.empty or y_train.empty:
            if verbose:
                print("LightGBM 澶辫触: 璁粌鏁版嵁涓虹┖")
            else:
                print("LightGBM 澶辫触: 璁粌鏁版嵁涓虹┖")
            return None
            
        X_test, _ = fe.make_features_for_prediction(pd.DataFrame(test), test_exog)
        params = {
            "n_estimators": 300,
            "max_depth": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_samples": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
        if regime["zero_heavy"]:
            params.update({"num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 15})
        model = LGBMRegressor(**params)
        fit_kwargs = {"verbose": -1}
        if len(X_train) >= 20:
            split = max(4, int(len(X_train) * 0.2))
            if len(X_train) - split >= 12:
                X_fit, X_valid = X_train.iloc[:-split], X_train.iloc[-split:]
                y_fit, y_valid = y_train.iloc[:-split], y_train.iloc[-split:]
                fit_kwargs.update({"eval_set": [(X_valid, y_valid)]})
                model.fit(X_fit, y_fit, **fit_kwargs)
            else:
                model.fit(X_train, y_train, **fit_kwargs)
        else:
            model.fit(X_train, y_train, **fit_kwargs)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)
        if regime["zero_heavy"]:
            y_pred = np.minimum(y_pred, max(regime["recent_mean"] * 1.5, 1.0))
        wmape = calculate_wmape(test.values, y_pred, min_non_zero_points=4)
        return {'name': 'LightGBM', 'wmape': wmape, 'preds': y_pred, 'model': model, 'params': model.get_params()}
    except Exception as e:
        if verbose:
            print(f"LightGBM 澶辫触: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"LightGBM 澶辫触: {e}")
        return None


def run_auto_arima(train, test, train_exog=None, test_exog=None, verbose=False, screening=None):
    try:
        regime = _infer_model_regime(train, screening)
        exog = train_exog.values if train_exog is not None else None
        seasonal = regime["history_weeks"] >= 52 and not regime["zero_heavy"]
        if regime["history_weeks"] >= 104:
            m = 52
        elif regime["history_weeks"] >= 26:
            m = 13
        else:
            m = 1
        model = pm.auto_arima(
            train.values,
            exogenous=exog,
            seasonal=seasonal and m > 1,
            m=m,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        test_exog_vals = test_exog.values if test_exog is not None else None
        y_pred = model.predict(n_periods=len(test), exogenous=test_exog_vals, return_conf_int=False)
        y_pred = np.maximum(y_pred, 0)
        if regime["zero_heavy"]:
            y_pred = np.minimum(y_pred, max(regime["recent_mean"] * 1.2, 1.0))
        wmape = calculate_wmape(test.values, y_pred, min_non_zero_points=4)
        return {'name': 'AutoARIMA', 'wmape': wmape, 'preds': y_pred, 'model': model, 'params': model.get_params()}
    except Exception as e:
        if verbose:
            print(f"AutoARIMA 澶辫触: {e}")
        else:
            print(f"AutoARIMA 澶辫触: {e}")
        return None


def _emit_model_log(log_fn, message):
    if log_fn is not None:
        log_fn(message)
    else:
        print(message)

def run_seasonal_naive(train, test, train_exog=None, test_exog=None, verbose=False, screening=None):
    try:
        clean = _series_to_float_series(train)
        if clean.empty:
            return None

        regime = _infer_model_regime(clean, screening)
        if regime["history_weeks"] >= 52:
            season_length = 52
        elif regime["history_weeks"] >= 26:
            season_length = 13
        else:
            season_length = max(4, min(8, regime["history_weeks"]))

        if season_length <= 1:
            forecast = np.full(len(test), float(clean.mean()) if len(clean) > 0 else 0.0)
        elif len(clean) >= season_length:
            tail = clean.iloc[-season_length:].to_numpy()
            forecast = np.resize(tail, len(test))
        else:
            forecast = np.full(len(test), float(clean.mean()) if len(clean) > 0 else 0.0)

        if regime["zero_heavy"]:
            forecast = np.minimum(forecast, max(regime["recent_mean"] * 1.3, 1.0))

        forecast = np.maximum(forecast, 0)
        wmape = calculate_wmape(test.values, forecast, min_non_zero_points=4)
        return {"name": "SeasonalNaive", "wmape": wmape, "preds": forecast, "model": None, "params": {"season_length": season_length}}
    except Exception as e:
        if verbose:
            print(f"SeasonalNaive 澶辫触: {e}")
        return None


def run_zero_aware_naive(train, test, train_exog=None, test_exog=None, verbose=False, screening=None):
    try:
        forecast = _build_zero_aware_forecast(train, len(test), screening)
        wmape = calculate_wmape(test.values, forecast, min_non_zero_points=4)
        return {
            "name": "ZeroAwareNaive",
            "wmape": wmape,
            "preds": forecast,
            "model": None,
            "params": {"strategy": "zero_aware"},
        }
    except Exception as e:
        if verbose:
            print(f"ZeroAwareNaive 澶辫触: {e}")
        return None


def _croston_sba_forecast(train_series, n_steps, alpha=0.15):
    values = np.asarray(train_series, dtype=float)
    values = np.maximum(values, 0.0)
    if values.size == 0:
        return np.zeros(n_steps)

    non_zero_idx = np.where(values > 0)[0]
    if non_zero_idx.size == 0:
        return np.zeros(n_steps)

    first_idx = int(non_zero_idx[0])
    z = float(values[first_idx])
    p = 1.0
    interval = 1

    for x in values[first_idx + 1 :]:
        if x > 0:
            z = z + alpha * (float(x) - z)
            p = p + alpha * (interval - p)
            interval = 1
        else:
            interval += 1

    demand_rate = z / p if p > 0 else 0.0
    sba_rate = max((1.0 - alpha / 2.0) * demand_rate, 0.0)
    return np.full(n_steps, sba_rate, dtype=float)


def run_croston_sba(train, test, train_exog=None, test_exog=None, verbose=False, screening=None):
    try:
        clean = _series_to_float_series(train)
        if clean.empty:
            return None
        regime = _infer_model_regime(clean, screening)
        forecast = _croston_sba_forecast(clean.values, len(test), alpha=0.15)
        if regime["zero_heavy"]:
            cap = max(regime["recent_mean"] * 1.2, 1.0)
            forecast = np.minimum(forecast, cap)
        forecast = np.maximum(forecast, 0)
        wmape = calculate_wmape(test.values, forecast, min_non_zero_points=4)
        return {
            "name": "CrostonSBA",
            "wmape": wmape,
            "preds": forecast,
            "model": None,
            "params": {"alpha": 0.15, "variant": "SBA"},
        }
    except Exception as e:
        if verbose:
            print(f"CrostonSBA 澶辫触: {e}")
        return None


def run_all_models(
    train,
    test,
    mode='smart',
    train_exog=None,
    test_exog=None,
    verbose=False,
    log_fn=None,
    screening=None,
    model_policy='standard',
):
    """
    运行所有预测模型，并记录每个模型的效果。
    """
    models = []
    conservative = str(model_policy).lower() == "conservative"
    enabled_base_models = []

    _emit_model_log(log_fn, f"\n模型竞赛启动 (mode={mode})...")
    _emit_model_log(log_fn, f"模型策略: {'conservative' if conservative else 'standard'}")
    _emit_model_log(log_fn, "=" * 70)

    if conservative:
        _emit_model_log(log_fn, "跳过 Prophet（conservative 策略）")
    else:
        enabled_base_models.append("Prophet")
        prophet_ready, prophet_reason = _check_prophet_runtime()
        if not prophet_ready:
            _emit_model_log(log_fn, f"Prophet: 跳过（运行环境不可用: {prophet_reason}）")
        else:
            _emit_model_log(log_fn, "运行 Prophet...")
            prophet_result = run_prophet(train, test, train_exog, test_exog, verbose, screening=screening)
            if prophet_result:
                models.append(prophet_result)
                _emit_model_log(log_fn, f"Prophet: WMAPE={prophet_result['wmape']:.2%}")
            else:
                _emit_model_log(log_fn, "Prophet: 失败")

    if conservative:
        _emit_model_log(log_fn, "跳过 XGBoost（conservative 策略）")
    else:
        enabled_base_models.append("XGBoost")
        _emit_model_log(log_fn, "运行 XGBoost...")
        xgboost_result = run_xgboost(train, test, train_exog, test_exog, verbose, screening=screening)
        if xgboost_result:
            models.append(xgboost_result)
            _emit_model_log(log_fn, f"XGBoost: WMAPE={xgboost_result['wmape']:.2%}")
        else:
            _emit_model_log(log_fn, "XGBoost: 失败")

    if conservative:
        _emit_model_log(log_fn, "跳过 LightGBM（conservative 策略）")
    else:
        enabled_base_models.append("LightGBM")
        _emit_model_log(log_fn, "运行 LightGBM...")
        lightgbm_result = run_lightgbm(train, test, train_exog, test_exog, verbose, screening=screening)
        if lightgbm_result:
            models.append(lightgbm_result)
            _emit_model_log(log_fn, f"LightGBM: WMAPE={lightgbm_result['wmape']:.2%}")
        else:
            _emit_model_log(log_fn, "LightGBM: 失败")

    enabled_base_models.append("AutoARIMA")
    _emit_model_log(log_fn, "运行 AutoARIMA...")
    autoarima_result = run_auto_arima(train, test, train_exog, test_exog, verbose, screening=screening)
    if autoarima_result:
        models.append(autoarima_result)
        _emit_model_log(log_fn, f"AutoARIMA: WMAPE={autoarima_result['wmape']:.2%}")
    else:
        _emit_model_log(log_fn, "AutoARIMA: 失败")

    models = [m for m in models if m is not None]
    base_results = {m['name']: m for m in models}

    _emit_model_log(log_fn, f"\n基础模型运行完成: {len(models)}/{len(enabled_base_models)} 个成功")

    if len(models) >= 2:
        _emit_model_log(log_fn, "运行融合算法...")

        avg_forecast = np.mean([m['preds'] for m in models], axis=0)
        avg_wmape = calculate_wmape(test, avg_forecast, min_non_zero_points=4)
        models.append({
            'name': 'Ensemble-Avg',
            'preds': avg_forecast,
            'wmape': avg_wmape,
            'model': None
        })
        _emit_model_log(log_fn, f"Ensemble-Avg: WMAPE={avg_wmape:.2%}")

        weights = [1 / m['wmape'] if np.isfinite(m['wmape']) and m['wmape'] > 0 else 0 for m in models if 'preds' in m]
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]
            base_models_for_weighted = [m for m in models if m['name'] not in ['Ensemble-Avg', 'Ensemble-Weighted']]
            weighted_forecast = np.average(
                [m['preds'] for m in base_models_for_weighted],
                axis=0,
                weights=weights[:len(base_models_for_weighted)],
            )
            weighted_wmape = calculate_wmape(test, weighted_forecast, min_non_zero_points=4)
            models.append({
                'name': 'Ensemble-Weighted',
                'preds': weighted_forecast,
                'wmape': weighted_wmape,
                'model': None
            })
            _emit_model_log(log_fn, f"Ensemble-Weighted: WMAPE={weighted_wmape:.2%}")

        _emit_model_log(log_fn, "融合算法运行完成")
    else:
        _emit_model_log(log_fn, "基础模型不足 2 个，跳过融合算法")

    zero_naive_result = run_zero_aware_naive(train, test, train_exog, test_exog, verbose, screening=screening)
    if zero_naive_result:
        models.append(zero_naive_result)
        _emit_model_log(log_fn, f"ZeroAwareNaive: WMAPE={zero_naive_result['wmape']:.2%}")

    croston_result = run_croston_sba(train, test, train_exog, test_exog, verbose, screening=screening)
    if croston_result:
        models.append(croston_result)
        _emit_model_log(log_fn, f"CrostonSBA: WMAPE={croston_result['wmape']:.2%}")

    seasonal_naive_result = run_seasonal_naive(train, test, train_exog, test_exog, verbose, screening=screening)
    if seasonal_naive_result:
        models.append(seasonal_naive_result)
        _emit_model_log(log_fn, f"SeasonalNaive: WMAPE={seasonal_naive_result['wmape']:.2%}")

    _emit_model_log(log_fn, "=" * 70)

    return models, base_results


def calculate_dynamic_shares(df_spu_idx, spu, spu_sales_weekly, future_dates):
    return _get_forecast_kernel().calculate_dynamic_shares(df_spu_idx, spu, spu_sales_weekly, future_dates)


def process_single_spu(spu, df_all, mode='smart', exog_cols=None, collect_viz=False, verbose=True, log_fn=None):
    result_df, message, _viz, profile = _get_forecast_kernel().process_single_spu(
        spu,
        df_all,
        mode=mode,
        exog_cols=exog_cols,
        collect_viz=collect_viz,
        verbose=verbose,
        log_fn=log_fn,
    )
    error = None if result_df is not None else message
    return result_df, error, profile


def calculate_spu_accuracy(actual, pred):
    actual_series = pd.Series(actual)
    pred_series = pd.Series(pred).reindex(actual_series.index)
    errors = actual_series - pred_series
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    non_zero = actual_series.replace(0, np.nan)
    mape = float(np.nanmean(np.abs(errors) / non_zero)) if non_zero.notna().any() else 0.0
    return {
        'wmape': float(calculate_wmape(actual_series, pred_series)),
        'mape': 0.0 if np.isnan(mape) else mape,
        'mae': mae,
        'rmse': rmse,
    }


def calculate_sku_accuracy(actual, pred):
    actual_df = pd.DataFrame(actual)
    pred_df = pd.DataFrame(pred).reindex(index=actual_df.index, columns=actual_df.columns, fill_value=0)
    sku_metrics = {}
    total_actual = float(actual_df.to_numpy().sum())
    weighted_error = 0.0

    for sku in actual_df.columns:
        sku_actual = actual_df[sku]
        sku_pred = pred_df[sku]
        wmape = float(calculate_wmape(sku_actual, sku_pred))
        total_sales = float(sku_actual.sum())
        weight = (total_sales / total_actual) if total_actual > 0 else 0.0
        weighted_error += wmape * weight
        sku_metrics[sku] = {
            'wmape': wmape,
            'total_sales': total_sales,
            'weight_in_spu': weight,
        }

    return {
        'wmape': weighted_error,
        'sku_metrics': sku_metrics,
        'sku_accuracy_json': json.dumps(sku_metrics, ensure_ascii=False),
    }


def predict_future(series, winner, n_steps, exog_series=None, future_exog=None, base_results=None):
    if winner['name'] == 'Prophet':
        df = pd.DataFrame({'ds': series.index, 'y': series.values})
        if exog_series is not None:
            # 纭繚澶栫敓鍙橀噺鐨勬椂闂寸储寮曚笌series鐨勬椂闂寸储寮曚竴鑷?            exog_series = exog_series.reindex(series.index)
            for col in exog_series.columns:
                df[col] = exog_series[col].values
        model = Prophet(
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
            changepoint_range=0.8
        )
        if exog_series is not None:
            for col in exog_series.columns:
                model.add_regressor(col)
        model.fit(df)
        future = model.make_future_dataframe(periods=n_steps, freq='W')
        if future_exog is not None:
            for col in future_exog.columns:
                if exog_series is not None:
                    hist_exog = exog_series[col].values
                    fut_exog = future_exog[col].values
                    all_exog = np.concatenate([hist_exog, fut_exog])
                    if len(all_exog) == len(future):
                        future[col] = all_exog
                    else:
                        future[col] = np.concatenate([np.full(len(series), np.nan), fut_exog])
                else:
                    future[col] = np.concatenate([np.full(len(series), np.nan), future_exog[col].values])
        forecast = model.predict(future)
        return np.maximum(forecast['yhat'].values[-n_steps:], 0)
    elif winner['name'] in ['XGBoost', 'LightGBM']:
        fe = FeatureEngineer()
        X, y = fe.make_features(pd.DataFrame(series), exog_series)
        model = winner['model']
        model.fit(X, y)
        future_dates = pd.date_range(series.index[-1], periods=n_steps + 1, freq='W')[1:]
        future_df = pd.DataFrame(index=future_dates, columns=['y'])
        future_df['y'] = 0
        if future_exog is not None:
            X_future, _ = fe.make_features_for_prediction(future_df, future_exog)
        else:
            X_future, _ = fe.make_features_for_prediction(future_df)
        preds = model.predict(X_future)
        return np.maximum(preds, 0)
    elif winner['name'] == 'SeasonalNaive':
        clean = _series_to_float_series(series)
        if clean.empty:
            return np.zeros(n_steps)
        if len(clean) >= 52:
            season_length = 52
        elif len(clean) >= 26:
            season_length = 13
        else:
            season_length = max(4, min(8, len(clean)))
        tail = clean.iloc[-season_length:].to_numpy() if len(clean) >= season_length else np.full(n_steps, float(clean.mean()))
        preds = np.resize(tail, n_steps)
        return np.maximum(preds, 0)
    elif winner['name'] == 'ZeroAwareNaive':
        return _build_zero_aware_forecast(series, n_steps, None)
    else:
        return np.array([series.mean()] * n_steps)


def safe_predictions(preds, fallback_value, model_name, history_series=None):
    preds = np.asarray(preds).flatten().copy()
    nan_count = np.sum(np.isnan(preds))
    if nan_count > 0:
        preds[np.isnan(preds)] = fallback_value
    preds = np.maximum(preds, 0)
    if history_series is not None and len(history_series) > 0:
        recent = pd.Series(history_series).tail(12).astype(float)
        recent_non_zero = recent[recent > 0]
        recent_anchor = float(recent_non_zero.median()) if not recent_non_zero.empty else float(fallback_value)
        recent_peak = float(recent.max()) if not recent.empty else float(fallback_value)
        recent_std = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0
        upper_cap = max(
            recent_anchor * 3.0,
            recent_peak * 2.0,
            recent_anchor + recent_std * 3.0,
            float(fallback_value) * 4.0,
            1.0,
        )
        preds = np.minimum(preds, upper_cap)
        if len(preds) > 1 and recent_anchor > 0:
            recent_changes = np.abs(np.diff(recent.to_numpy())) if len(recent) > 1 else np.array([0.0])
            max_step = max(float(np.nanmedian(recent_changes)) * 3.0, recent_std * 2.0, recent_anchor * 0.8, 1.0)
            smoothed = preds.copy()
            for idx in range(1, len(smoothed)):
                if preds[idx] <= 0:
                    smoothed[idx] = 0.0
                    continue
                lower_bound = max(smoothed[idx - 1] - max_step, 0.0)
                upper_bound_step = smoothed[idx - 1] + max_step
                smoothed[idx] = float(np.clip(smoothed[idx], lower_bound, min(upper_cap, upper_bound_step)))
            preds = smoothed
    return preds


def extract_seasonal_factors_52week(series, period=52):
    try:
        if len(series) < period:
            factors = {f'week_{i+1}': 1.0 for i in range(period)}
            return json.dumps(factors, ensure_ascii=False)
        df = pd.DataFrame({'ds': series.index, 'y': series.values})
        model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        model.fit(df)
        forecast = model.predict(df)
        seasonal = forecast['yearly'].values
        seasonal = seasonal - seasonal.mean() + 1
        seasonal = np.clip(seasonal, 0.5, 1.5)
        factors = {f'week_{i+1}': float(seasonal[i % period]) for i in range(period)}
        return json.dumps(factors, ensure_ascii=False)
    except Exception as e:
        factors = {f'week_{i+1}': 1.0 for i in range(period)}
        return json.dumps(factors, ensure_ascii=False)


def clean_params_for_db(params):
    if not params:
        return "{}"
    try:
        def clean_value(v):
            if isinstance(v, (np.ndarray, np.generic)):
                return v.tolist()
            elif isinstance(v, (pd.Series, pd.DataFrame)):
                return v.to_dict()
            elif isinstance(v, (list, dict, str, int, float, bool, type(None))):
                return v
            else:
                return str(v)
        clean_params = {k: clean_value(v) for k, v in params.items()}
        return json.dumps(clean_params, ensure_ascii=False)
    except:
        return "{}"


def plot_best_spu_style(profile, train, test, results, future, future_dates, sku_future_df=None, save_path=None, show_plot=True):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    try:
        font_path = 'static/file-ngwyeoEN29l1M3O1QpdxCwkj-sider-font.ttf'
        if os.path.exists(font_path):
            font_prop = FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
    except:
        pass
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    all_dates = pd.date_range(train.index[0], future_dates[-1], freq='W')
    ax1.plot(train.index, train.values, 'gray', label='鍘嗗彶鏁版嵁', linewidth=2)
    ax1.plot(test.index, test.values, 'k-', marker='o', label='Actual', linewidth=2, markersize=6)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        ax1.plot(test.index, res['preds'], color, marker='s', label=f'{res["name"]} (WMAPE: {res["wmape"]:.2%})', linewidth=1.5, markersize=4)
    ax1.plot(future_dates, future, 'purple', marker='d', label=f'{profile.winner_algo} 棰勬祴', linewidth=3, markersize=6)
    if sku_future_df is not None and not sku_future_df.empty:
        top_skus = sku_future_df.sum().nlargest(5).index
        for i, sku in enumerate(top_skus):
            color = colors[(i + 1) % len(colors)]
            ax1.plot(future_dates, sku_future_df[sku], color, linestyle='--', marker='o', label=f'SKU {sku}', linewidth=1.5, markersize=4)
    val_start = test.index[0]
    val_end = test.index[-1]
    ax1.axvspan(val_start, val_end, alpha=0.2, color='yellow', label='楠岃瘉鍖洪棿')
    pred_start = future_dates[0]
    pred_end = future_dates[-1]
    ax1.axvspan(pred_start, pred_end, alpha=0.2, color='green', label='棰勬祴鍖洪棿')
    ax1.set_title(f"SPU {profile.spu} forecast - winner: {profile.winner_algo} (WMAPE: {profile.winner_wmape:.2%})", fontsize=16)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Sales", fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)
    model_names = [r['name'] for r in results]
    wmapes = [r['wmape'] for r in results]
    winner_idx = model_names.index(profile.winner_algo)
    colors = ['#1f77b4'] * len(model_names)
    colors[winner_idx] = '#d62728'
    ax2.bar(model_names, wmapes, color=colors)
    ax2.set_title("Model WMAPE Comparison", fontsize=14)
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("WMAPE", fontsize=12)
    for i, wmape in enumerate(wmapes):
        ax2.text(i, wmape, f'{wmape:.2%}', ha='center', va='bottom', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   馃搳 鍥捐〃宸蹭繚瀛樿嚦: {save_path}")
    if show_plot:
        plt.show()
    plt.close()

