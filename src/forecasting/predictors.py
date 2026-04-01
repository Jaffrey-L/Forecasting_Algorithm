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
from functools import partial
from src.forecasting.models import *

forecast_kernel = None


def _get_forecast_kernel():
    if forecast_kernel is not None:
        return forecast_kernel
    from src.forecasting import kernel as loaded_kernel

    return loaded_kernel


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


def calculate_wmape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 确保y_true和y_pred长度一致
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.sum(np.abs(y_true[mask] - y_pred[mask])) / np.sum(np.abs(y_true[mask]))


def run_prophet(train, test, train_exog=None, test_exog=None, verbose=False):
    try:
        df_train = pd.DataFrame({'ds': train.index, 'y': train.values})
        
        # 检查df_train是否为空
        if df_train.empty:
            if verbose:
                print("Prophet 失败: 训练数据为空")
            else:
                print("Prophet 失败: 训练数据为空")
            return None
            
        if train_exog is not None:
            # 检查train_exog是否为空
            if not train_exog.empty:
                # 确保train_exog的索引与train一致
                train_exog_aligned = train_exog.reindex(train.index)
                # 只添加存在的列
                for col in train_exog_aligned.columns:
                    if not train_exog_aligned[col].isna().all():
                        df_train[col] = train_exog_aligned[col].values
        model = Prophet(
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0,
            changepoint_range=0.8
        )
        if train_exog is not None:
            # 检查train_exog是否为空
            if not train_exog.empty:
                # 只添加存在的列
                for col in train_exog.columns:
                    model.add_regressor(col)
        model.fit(df_train)
        df_test = pd.DataFrame({'ds': test.index})
        if test_exog is not None:
            # 检查test_exog是否为空
            if not test_exog.empty:
                # 确保test_exog的索引与test一致
                test_exog_aligned = test_exog.reindex(test.index)
                # 只添加存在的列
                for col in test_exog_aligned.columns:
                    if not test_exog_aligned[col].isna().all():
                        df_test[col] = test_exog_aligned[col].values
        forecast = model.predict(df_test)
        y_pred = forecast['yhat'].values
        y_pred = np.maximum(y_pred, 0)
        wmape = calculate_wmape(test.values, y_pred)
        return {'name': 'Prophet', 'wmape': wmape, 'preds': y_pred, 'model': model, 'params': model.params if hasattr(model, 'params') else {}}
    except Exception as e:
        if verbose:
            print(f"Prophet 失败: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"Prophet 失败: {e}")
        return None


def run_xgboost(train, test, train_exog=None, test_exog=None, verbose=False):
    try:
        fe = FeatureEngineer()
        X_train, y_train = fe.make_features(pd.DataFrame(train), train_exog)
        
        # 检查X_train或y_train是否为空
        if X_train.empty or y_train.empty:
            if verbose:
                print("XGBoost 失败: 训练数据为空")
            else:
                print("XGBoost 失败: 训练数据为空")
            return None
            
        X_test, _ = fe.make_features_for_prediction(pd.DataFrame(test), test_exog)
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)
        wmape = calculate_wmape(test.values, y_pred)
        return {'name': 'XGBoost', 'wmape': wmape, 'preds': y_pred, 'model': model, 'params': model.get_params()}
    except Exception as e:
        if verbose:
            print(f"XGBoost 失败: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"XGBoost 失败: {e}")
        return None


def run_lightgbm(train, test, train_exog=None, test_exog=None, verbose=False):
    try:
        fe = FeatureEngineer()
        X_train, y_train = fe.make_features(pd.DataFrame(train), train_exog)
        
        # 检查X_train或y_train是否为空
        if X_train.empty or y_train.empty:
            if verbose:
                print("LightGBM 失败: 训练数据为空")
            else:
                print("LightGBM 失败: 训练数据为空")
            return None
            
        X_test, _ = fe.make_features_for_prediction(pd.DataFrame(test), test_exog)
        model = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)
        wmape = calculate_wmape(test.values, y_pred)
        return {'name': 'LightGBM', 'wmape': wmape, 'preds': y_pred, 'model': model, 'params': model.get_params()}
    except Exception as e:
        if verbose:
            print(f"LightGBM 失败: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"LightGBM 失败: {e}")
        return None


def run_auto_arima(train, test, train_exog=None, test_exog=None, verbose=False):
    try:
        exog = train_exog.values if train_exog is not None else None
        model = pm.auto_arima(
            train.values,
            exogenous=exog,
            seasonal=True,
            m=52,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        test_exog_vals = test_exog.values if test_exog is not None else None
        y_pred, _ = model.predict(n_periods=len(test), exogenous=test_exog_vals, return_conf_int=False)
        y_pred = np.maximum(y_pred, 0)
        wmape = calculate_wmape(test.values, y_pred)
        return {'name': 'AutoARIMA', 'wmape': wmape, 'preds': y_pred, 'model': model, 'params': model.get_params()}
    except Exception as e:
        if verbose:
            print(f"AutoARIMA 失败: {e}")
        else:
            print(f"AutoARIMA 失败: {e}")
        return None


def run_all_models(train, test, mode='smart', train_exog=None, test_exog=None, verbose=False):
    """
    运行所有预测模型，包括4种基础模型和2种融合算法
    
    基础模型：
    1. Prophet
    2. XGBoost
    3. LightGBM
    4. AutoARIMA
    
    融合算法：
    5. Ensemble-Avg (简单平均融合)
    6. Ensemble-Weighted (加权融合)
    """
    models = []
    
    print(f"\n⚡ 启动模型竞赛 (mode={mode})...")
    print("=" * 70)
    
    # 运行基础模型 - 所有模式都运行全部4种基础模型
    print("🔄 运行 Prophet...")
    prophet_result = run_prophet(train, test, train_exog, test_exog, verbose)
    if prophet_result:
        models.append(prophet_result)
        print(f"   ✅ Prophet: WMAPE={prophet_result['wmape']:.2%}")
    else:
        print(f"   ❌ Prophet: 失败")
    
    print("🔄 运行 XGBoost...")
    xgboost_result = run_xgboost(train, test, train_exog, test_exog, verbose)
    if xgboost_result:
        models.append(xgboost_result)
        print(f"   ✅ XGBoost: WMAPE={xgboost_result['wmape']:.2%}")
    else:
        print(f"   ❌ XGBoost: 失败")
    
    print("🔄 运行 LightGBM...")
    lightgbm_result = run_lightgbm(train, test, train_exog, test_exog, verbose)
    if lightgbm_result:
        models.append(lightgbm_result)
        print(f"   ✅ LightGBM: WMAPE={lightgbm_result['wmape']:.2%}")
    else:
        print(f"   ❌ LightGBM: 失败")
    
    print("🔄 运行 AutoARIMA...")
    autoarima_result = run_auto_arima(train, test, train_exog, test_exog, verbose)
    if autoarima_result:
        models.append(autoarima_result)
        print(f"   ✅ AutoARIMA: WMAPE={autoarima_result['wmape']:.2%}")
    else:
        print(f"   ❌ AutoARIMA: 失败")
    
    # 过滤掉失败的模型
    models = [m for m in models if m is not None]
    base_results = {m['name']: m for m in models}
    
    print(f"\n📊 基础模型运行完成: {len(models)}/4 个成功")
    
    # 运行融合算法 - 需要至少2个基础模型成功
    if len(models) >= 2:
        print("\n🔄 运行融合算法...")
        
        # 简单平均融合
        avg_forecast = np.mean([m['preds'] for m in models], axis=0)
        avg_wmape = calculate_wmape(test, avg_forecast)
        models.append({
            'name': 'Ensemble-Avg',
            'preds': avg_forecast,
            'wmape': avg_wmape,
            'model': None
        })
        print(f"   ✅ Ensemble-Avg: WMAPE={avg_wmape:.2%}")
        
        # 加权融合（基于WMAPE倒数）
        weights = [1/m['wmape'] if m['wmape'] > 0 else 0 for m in models if 'preds' in m]
        if sum(weights) > 0:
            weights = [w/sum(weights) for w in weights]
            # 只对基础模型进行加权融合
            base_models_for_weighted = [m for m in models if m['name'] not in ['Ensemble-Avg', 'Ensemble-Weighted']]
            weighted_forecast = np.average([m['preds'] for m in base_models_for_weighted], axis=0, weights=weights[:len(base_models_for_weighted)])
            weighted_wmape = calculate_wmape(test, weighted_forecast)
            models.append({
                'name': 'Ensemble-Weighted',
                'preds': weighted_forecast,
                'wmape': weighted_wmape,
                'model': None
            })
            print(f"   ✅ Ensemble-Weighted: WMAPE={weighted_wmape:.2%}")
        
        print(f"\n🎉 融合算法运行完成")
    else:
        print("\n⚠️ 基础模型不足2个，跳过融合算法")
    
    print("=" * 70)
    
    return models, base_results


def calculate_dynamic_shares(df_spu_idx, spu, spu_sales_weekly, future_dates):
    return _get_forecast_kernel().calculate_dynamic_shares(df_spu_idx, spu, spu_sales_weekly, future_dates)


def process_single_spu(spu, df_all, mode='smart', exog_cols=None, collect_viz=False, verbose=True):
    result_df, message, _viz, profile = _get_forecast_kernel().process_single_spu(
        spu,
        df_all,
        mode=mode,
        exog_cols=exog_cols,
        collect_viz=collect_viz,
        verbose=verbose,
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
            # 确保外生变量的时间索引与series的时间索引一致
            exog_series = exog_series.reindex(series.index)
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
            # 为future数据框添加外生变量
            # 历史部分使用exog_series，未来部分使用future_exog
            for col in future_exog.columns:
                if exog_series is not None:
                    # 历史外生变量
                    hist_exog = exog_series[col].values
                    # 未来外生变量
                    fut_exog = future_exog[col].values
                    # 合并历史和未来外生变量
                    all_exog = np.concatenate([hist_exog, fut_exog])
                    # 确保长度匹配
                    if len(all_exog) == len(future):
                        future[col] = all_exog
                    else:
                        # 如果长度不匹配，只使用未来外生变量
                        future[col] = np.concatenate([np.full(len(series), np.nan), fut_exog])
                else:
                    # 如果没有历史外生变量，只使用未来外生变量
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
    else:
        return np.array([series.mean()] * n_steps)


def safe_predictions(preds, fallback_value, model_name):
    preds = np.asarray(preds).flatten().copy()
    nan_count = np.sum(np.isnan(preds))
    if nan_count > 0:
        preds[np.isnan(preds)] = fallback_value
    preds = np.maximum(preds, 0)
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
    ax1.plot(train.index, train.values, 'gray', label='历史数据', linewidth=2)
    ax1.plot(test.index, test.values, 'k-', marker='o', label='实际值', linewidth=2, markersize=6)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        ax1.plot(test.index, res['preds'], color, marker='s', label=f'{res["name"]} (WMAPE: {res["wmape"]:.2%})', linewidth=1.5, markersize=4)
    ax1.plot(future_dates, future, 'purple', marker='d', label=f'{profile.winner_algo} 预测', linewidth=3, markersize=6)
    if sku_future_df is not None and not sku_future_df.empty:
        top_skus = sku_future_df.sum().nlargest(5).index
        for i, sku in enumerate(top_skus):
            color = colors[(i + 1) % len(colors)]
            ax1.plot(future_dates, sku_future_df[sku], color, linestyle='--', marker='o', label=f'SKU {sku}', linewidth=1.5, markersize=4)
    val_start = test.index[0]
    val_end = test.index[-1]
    ax1.axvspan(val_start, val_end, alpha=0.2, color='yellow', label='验证区间')
    pred_start = future_dates[0]
    pred_end = future_dates[-1]
    ax1.axvspan(pred_start, pred_end, alpha=0.2, color='green', label='预测区间')
    ax1.set_title(f'SPU {profile.spu} 销售预测 - 胜出模型: {profile.winner_algo} (WMAPE: {profile.winner_wmape:.2%})', fontsize=16)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('销量', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)
    model_names = [r['name'] for r in results]
    wmapes = [r['wmape'] for r in results]
    winner_idx = model_names.index(profile.winner_algo)
    colors = ['#1f77b4'] * len(model_names)
    colors[winner_idx] = '#d62728'
    ax2.bar(model_names, wmapes, color=colors)
    ax2.set_title('模型 WMAPE 对比', fontsize=14)
    ax2.set_xlabel('模型', fontsize=12)
    ax2.set_ylabel('WMAPE', fontsize=12)
    for i, wmape in enumerate(wmapes):
        ax2.text(i, wmape, f'{wmape:.2%}', ha='center', va='bottom', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 图表已保存至: {save_path}")
    if show_plot:
        plt.show()
    plt.close()
