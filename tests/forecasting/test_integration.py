"""
集成测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


class TestIntegration:
    """集成测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """生成示例数据"""
        dates = pd.date_range('2025-01-01', periods=100, freq='W')
        
        # SPU数据
        spu_data = {
            'date': dates,
            'spu': ['SPU001'] * 100,
            'sales': np.random.randint(100, 1000, 100),
            'price': np.random.uniform(10, 50, 100),
            'ad_cost': np.random.uniform(100, 500, 100)
        }
        df_spu = pd.DataFrame(spu_data)
        
        # SKU数据
        dates_all = dates * 3
        sku_data = {
            'date': dates_all,
            'spu': ['SPU001'] * 300,
            'sku': ['SKU001', 'SKU002', 'SKU003'] * 100,
            'sales': np.random.randint(10, 100, 300),
            'price': np.random.uniform(10, 50, 300)
        }
        df_sku = pd.DataFrame(sku_data)
        
        # 合并数据
        df_all = pd.concat([df_spu, df_sku], ignore_index=True)
        
        return df_all
    
    @patch('src.forecasting.predictors.run_all_models')
    @patch('src.forecasting.predictors.calculate_dynamic_shares')
    def test_spu_prediction_pipeline(self, mock_dynamic_shares, mock_run_models, sample_data):
        """测试SPU预测流程"""
        from src.forecasting.predictors import process_single_spu
        
        # 模拟模型运行结果
        mock_run_models.return_value = [
            {
                'algo': 'Prophet',
                'wmape': 0.15,
                'forecast': np.random.randint(100, 1000, 10),
                'params': {'param1': 'value1'}
            }
        ]
        
        # 模拟动态份额计算
        mock_dynamic_shares.return_value = (
            ['{"SKU001":0.3,"SKU002":0.4,"SKU003":0.3}'] * 10,
            pd.DataFrame({
                'SKU001': [0.3] * 10,
                'SKU002': [0.4] * 10,
                'SKU003': [0.3] * 10
            })
        )
        
        # 运行预测
        result, error, profile = process_single_spu('SPU001', sample_data, verbose=False)
        
        # 验证结果
        assert result is not None
        assert error is None
        assert 'forecast' in result
        assert 'algo' in result
        assert 'wmape' in result
    
    @patch('src.forecasting.predictors.process_single_spu')
    def test_full_forecast_pipeline(self, mock_process_spu, sample_data):
        """测试完整预测流程"""
        from src.forecasting.main import main
        
        # 模拟SPU处理结果
        mock_process_spu.return_value = (
            {
                'forecast': np.random.randint(100, 1000, 10),
                'algo': 'Prophet',
                'wmape': 0.15,
                'validation_wmape': 0.12,
                'best_params': {'param1': 'value1'},
                'sku_accuracy_json': '{"SKU001":0.1,"SKU002":0.2,"SKU003":0.3}',
                'sku_share_json': '{"SKU001":0.3,"SKU002":0.4,"SKU003":0.3}'
            },
            None,
            {
                'spu': 'SPU001',
                'data_quality': 'good',
                'model_recommendation': 'Prophet',
                'training_weeks': 52
            }
        )
        
        # 运行主程序
        result = main(df_all=sample_data, spu_list=['SPU001'], verbose=False)
        
        # 验证结果
        assert result is not None
        assert isinstance(result, dict)
        assert 'SPU001' in result
        assert result['SPU001']['status'] == 'success'
    
    @patch('src.forecasting.monitor.create_engine')
    def test_monitoring_pipeline(self, mock_create_engine, sample_data):
        """测试监控流程"""
        from src.forecasting.monitor import ForecastingMonitor
        
        # 模拟数据库连接
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        # 模拟查询结果
        mock_conn.execute.return_value = sample_data
        
        # 创建监控器
        monitor = ForecastingMonitor('postgresql://test:test@localhost:5432/test_db', wmape_threshold=0.30)
        
        # 运行监控
        result = monitor.run_monitoring(days=30)
        
        # 验证结果
        assert 'summary' in result
        assert 'stats' in result
        assert 'anomalies' in result
        assert 'trends' in result
        assert 'alerts' in result
        
        # 验证摘要
        summary = result['summary']
        assert summary['total_records'] > 0
        assert summary['threshold_exceeded_count'] >= 0
        assert summary['anomaly_count'] >= 0
        assert summary['alert_count'] >= 0
    
    @patch('src.forecasting.predictors.process_single_spu')
    @patch('src.forecasting.monitor.create_engine')
    def test_end_to_end_workflow(self, mock_create_engine, mock_process_spu, sample_data):
        """测试端到端工作流"""
        from src.forecasting.main import main
        from src.forecasting.monitor import ForecastingMonitor
        
        # 模拟SPU处理结果
        mock_process_spu.return_value = (
            {
                'forecast': np.random.randint(100, 1000, 10),
                'algo': 'Prophet',
                'wmape': 0.15,
                'validation_wmape': 0.12,
                'best_params': {'param1': 'value1'},
                'sku_accuracy_json': '{"SKU001":0.1,"SKU002":0.2,"SKU003":0.3}',
                'sku_share_json': '{"SKU001":0.3,"SKU002":0.4,"SKU003":0.3}'
            },
            None,
            {
                'spu': 'SPU001',
                'data_quality': 'good',
                'model_recommendation': 'Prophet',
                'training_weeks': 52
            }
        )
        
        # 模拟数据库连接
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        # 模拟查询结果
        mock_conn.execute.return_value = sample_data
        
        # 步骤1: 运行预测
        forecast_result = main(df_all=sample_data, spu_list=['SPU001'], verbose=False)
        assert forecast_result is not None
        
        # 步骤2: 运行监控
        monitor = ForecastingMonitor('postgresql://test:test@localhost:5432/test_db', wmape_threshold=0.30)
        monitor_result = monitor.run_monitoring(days=30)
        assert monitor_result is not None
        
        # 步骤3: 验证结果
        assert 'SPU001' in forecast_result
        assert forecast_result['SPU001']['status'] == 'success'
        assert monitor_result['summary']['total_records'] > 0
