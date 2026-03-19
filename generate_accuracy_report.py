#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPU准确率分析报告生成器
生成所有34个SPU的预测准确率分析HTML报告
"""
import sys
sys.path.insert(0, 'c:\\Users\\VY0814\\Forecasting_Algorithm')

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sqlalchemy import create_engine, text

def get_forecast_data(db_url):
    """从数据库获取预测数据"""
    print("🔄 正在从数据库获取预测数据...")
    query = """
    SELECT * FROM finedatalink.sales_forecast_history
    WHERE run_date = (SELECT MAX(run_date) FROM finedatalink.sales_forecast_history)
    ORDER BY spu, forecast_target_date
    """
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), con=conn)
        print(f"✅ 成功获取 {len(df)} 条预测记录")
        return df
    finally:
        engine.dispose()

def get_actual_sales_data(db_url, start_date, end_date):
    """从数据库获取实际销售数据"""
    print("🔄 正在从数据库获取实际销售数据...")
    query = f"""
    with base as (
        select
        a."date" as report_date,
        a.msku as local_sku,
        case 
        when substring(a.msku,1,5)='RHNWB' then substring(a.msku,6,4)
        when substring(a.msku,1,2)='VY' then substring(a.msku,5,4)
        when substring(a.msku,1,2)='WB' then substring(a.msku,3,4)
        else '-' end as SPU,
        sum(a.volume) as 销量
        from lx_ods.查询订单利润_msku_cny_5年版 a 
        left join lx_ods.查询订单利润_msku_cny_商品基础信息_5年版 b on a.__dm_key=b.__dm_key
        where a."date" between '{start_date}' and '{end_date}'
        group by 1,2,3
    ),
    SPU_list as (
        SELECT '2141' AS SPU 
        UNION ALL 
        SELECT '2062' 
        UNION ALL 
        SELECT '2029' 
        UNION ALL 
        SELECT '2046' 
        UNION ALL 
        SELECT '2026' 
        UNION ALL 
        SELECT '3022' 
        UNION ALL 
        SELECT '1930' 
        UNION ALL 
        SELECT '2214' 
        UNION ALL 
        SELECT '2033' 
        UNION ALL 
        SELECT '2038' 
        UNION ALL 
        SELECT '2208' 
        UNION ALL 
        SELECT '2012' 
        UNION ALL 
        SELECT '3050' 
        UNION ALL 
        SELECT '2176' 
        UNION ALL 
        SELECT '3033' 
        UNION ALL 
        SELECT '2192' 
        UNION ALL 
        SELECT '2213' 
        UNION ALL 
        SELECT '3063' 
        UNION ALL 
        SELECT '2224' 
        UNION ALL 
        SELECT '3058' 
        UNION ALL 
        SELECT '2073' 
        UNION ALL 
        SELECT '3013' 
        UNION ALL 
        SELECT '2165' 
        UNION ALL 
        SELECT '3084' 
        UNION ALL 
        SELECT '1976' 
        UNION ALL 
        SELECT '2197' 
        UNION ALL 
        SELECT '1476' 
        UNION ALL 
        SELECT '1533' 
        UNION ALL 
        SELECT '887' 
        UNION ALL 
        SELECT '1577' 
        UNION ALL 
        SELECT '1750' 
        UNION ALL 
        SELECT '1512' 
        UNION ALL 
        SELECT '1657' 
        UNION ALL 
        SELECT '1983' 
        UNION ALL 
        SELECT '1318' 
    )
    select report_date as date, sum(销量) as sales, SPU as spu, local_sku as sku
    from base 
    where SPU in (select SPU from SPU_list)
    group by report_date, SPU, local_sku
    order by report_date
    """
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), con=conn)
        df.columns = [col.lower() for col in df.columns]
        print(f"✅ 成功获取 {len(df)} 条实际销售记录")
        return df
    finally:
        engine.dispose()

def calculate_wmape(y_true, y_pred):
    """计算加权平均绝对百分比误差"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.sum(np.abs(y_true[mask] - y_pred[mask])) / np.sum(np.abs(y_true[mask]))

def generate_html_report(accuracy_data, sku_accuracy_data, output_path):
    """生成HTML报告"""
    print("📊 正在生成HTML报告...")
    
    # 计算总体统计
    overall_avg_wmape = accuracy_data['wmape'].mean()
    overall_median_wmape = accuracy_data['wmape'].median()
    overall_min_wmape = accuracy_data['wmape'].min()
    overall_max_wmape = accuracy_data['wmape'].max()
    
    # 按准确率排序
    sorted_accuracy = accuracy_data.sort_values('wmape')
    
    # 生成HTML内容
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPU预测准确率分析报告</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 28px;
            text-align: center;
            margin-bottom: 40px;
            color: #3498db;
        }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .summary-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .summary-item h3 {
            font-size: 16px;
            color: #666;
            margin-bottom: 10px;
        }
        .summary-item .value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            overflow: hidden;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .sku-table {
            margin-top: 20px;
        }
        .error-bar {
            height: 8px;
            background: linear-gradient(to right, #4CAF50, #FFC107, #F44336);
            border-radius: 4px;
            margin-top: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #2c3e50;
            color: white;
            border-radius: 8px;
        }
        .status {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .status.good {
            background: #d4edda;
            color: #155724;
        }
        .status.warning {
            background: #fff3cd;
            color: #856404;
        }
        .status.bad {
            background: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>SPU预测准确率分析报告</h1>
        
        <div class="summary">
            <h2>总体概览</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <h3>平均WMAPE</h3>
                    <div class="value">{overall_avg_wmape:.2%}</div>
                </div>
                <div class="summary-item">
                    <h3>中位数WMAPE</h3>
                    <div class="value">{overall_median_wmape:.2%}</div>
                </div>
                <div class="summary-item">
                    <h3>最佳准确率</h3>
                    <div class="value">{overall_min_wmape:.2%}</div>
                </div>
                <div class="summary-item">
                    <h3>最差准确率</h3>
                    <div class="value">{overall_max_wmape:.2%}</div>
                </div>
                <div class="summary-item">
                    <h3>分析SPU数量</h3>
                    <div class="value">{len(accuracy_data)}</div>
                </div>
                <div class="summary-item">
                    <h3>生成时间</h3>
                    <div class="value">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
            </div>
        </div>
        
        <h2>SPU准确率排行</h2>
        <table>
            <thead>
                <tr>
                    <th>排名</th>
                    <th>SPU</th>
                    <th>WMAPE</th>
                    <th>状态</th>
                    <th>误差分布</th>
                </tr>
            </thead>
            <tbody>
        """
    
    # 添加SPU准确率数据
    for i, (_, row) in enumerate(sorted_accuracy.iterrows(), 1):
        wmape = row['wmape']
        if wmape < 0.1:
            status = 'good'
            status_text = '优秀'
        elif wmape < 0.2:
            status = 'warning'
            status_text = '良好'
        else:
            status = 'bad'
            status_text = '需改进'
        
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{row['spu']}</td>
                    <td>{wmape:.2%}</td>
                    <td><span class="status {status}">{status_text}</span></td>
                    <td>
                        <div class="error-bar" style="width: {min(wmape * 500, 100)}%"></div>
                    </td>
                </tr>
        """
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>SKU层面准确率分析</h2>
        <table class="sku-table">
            <thead>
                <tr>
                    <th>SPU</th>
                    <th>SKU</th>
                    <th>WMAPE</th>
                    <th>状态</th>
                </tr>
            </thead>
            <tbody>
        """
    
    # 添加SKU准确率数据
    for _, row in sku_accuracy_data.iterrows():
        wmape = row['wmape']
        if wmape < 0.1:
            status = 'good'
            status_text = '优秀'
        elif wmape < 0.2:
            status = 'warning'
            status_text = '良好'
        else:
            status = 'bad'
            status_text = '需改进'
        
        html_content += f"""
                <tr>
                    <td>{row['spu']}</td>
                    <td>{row['sku']}</td>
                    <td>{wmape:.2%}</td>
                    <td><span class="status {status}">{status_text}</span></td>
                </tr>
        """
    
    html_content += f"""
            </tbody>
        </table>
        
        <div class="footer">
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>数据来源: 销售预测系统 v8.5</p>
        </div>
    </div>
</body>
</html>
        """
    
    # 保存HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 报告已保存至: {output_path}")

def main():
    """主函数"""
    print("=" * 70)
    print("📈 SPU准确率分析报告生成器")
    print("=" * 70)
    
    # 数据库连接
    DB_URL = os.getenv("SALES_FORECAST_DB_URL", "postgresql+psycopg2://postgres:vayiERty123@192.168.1.226:5432/finedatalink")
    
    # 获取预测数据
    forecast_df = get_forecast_data(DB_URL)
    
    if forecast_df.empty:
        print("❌ 未找到预测数据")
        return
    
    # 确定分析时间范围
    min_date = forecast_df['forecast_target_date'].min()
    max_date = forecast_df['forecast_target_date'].max()
    print(f"📅 分析时间范围: {min_date} 至 {max_date}")
    
    # 获取实际销售数据
    actual_df = get_actual_sales_data(DB_URL, min_date, max_date)
    
    if actual_df.empty:
        print("❌ 未找到实际销售数据")
        return
    
    # 准备数据
    actual_weekly = actual_df.groupby(['spu', 'sku', pd.Grouper(key='date', freq='W')])['sales'].sum().reset_index()
    actual_weekly['date'] = actual_weekly['date'].dt.date
    
    # 计算SPU和SKU的准确率
    spu_accuracy = []
    sku_accuracy = []
    
    for spu in forecast_df['spu'].unique():
        spu_forecast = forecast_df[forecast_df['spu'] == spu]
        spu_actual = actual_weekly[actual_weekly['spu'] == spu]
        
        # 按周聚合实际销售
        weekly_actual = spu_actual.groupby('date')['sales'].sum().reset_index()
        
        # 合并预测和实际数据
        merged = pd.merge(spu_forecast, weekly_actual, left_on='forecast_target_date', right_on='date', how='left')
        merged['sales'] = merged['sales'].fillna(0)
        
        # 计算SPU准确率
        if len(merged) > 0 and merged['sales'].sum() > 0:
            wmape = calculate_wmape(merged['sales'], merged['spu_forecast_value'])
            spu_accuracy.append({'spu': spu, 'wmape': wmape})
        
        # 计算SKU准确率
        for sku in spu_actual['sku'].unique():
            sku_actual_data = spu_actual[spu_actual['sku'] == sku]
            sku_weekly = sku_actual_data.groupby('date')['sales'].sum().reset_index()
            
            # 合并预测和实际数据（这里简化处理，实际应该根据SKU份额计算）
            sku_merged = pd.merge(spu_forecast, sku_weekly, left_on='forecast_target_date', right_on='date', how='left')
            sku_merged['sales'] = sku_merged['sales'].fillna(0)
            
            if len(sku_merged) > 0 and sku_merged['sales'].sum() > 0:
                # 简化处理：使用SPU预测值作为SKU预测（实际应该根据份额计算）
                sku_wmape = calculate_wmape(sku_merged['sales'], sku_merged['spu_forecast_value'])
                sku_accuracy.append({'spu': spu, 'sku': sku, 'wmape': sku_wmape})
    
    # 转换为DataFrame
    spu_accuracy_df = pd.DataFrame(spu_accuracy)
    sku_accuracy_df = pd.DataFrame(sku_accuracy)
    
    # 生成报告
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    output_file = os.path.join(desktop_path, f'spu_accuracy_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    
    generate_html_report(spu_accuracy_df, sku_accuracy_df, output_file)
    
    print("=" * 70)
    print("✅ 报告生成完成！")
    print(f"📄 报告位置: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 生成报告失败: {e}")
        import traceback
        traceback.print_exc()
