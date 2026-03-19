import os
import datetime
import pandas as pd
from sqlalchemy import create_engine, text


def save_to_database(df_spu_level, db_url):
    print("\n🚀 正在将 SPU 级预测结果及 SKU 占比 JSON 写入数据库...")

    df_db = df_spu_level.copy()

    # ✅ 所有日期列强制转成 Python 原生 date，兼容 PostgreSQL DATE 类型
    for col in ['run_date', 'forecast_target_date', 'data_end_date']:
        if col in df_db.columns:
            df_db[col] = pd.to_datetime(df_db[col]).dt.date

    df_db['create_time']       = datetime.datetime.now()
    df_db['sku_share_json']    = df_db['sku_share_json'].astype(str)
    df_db['best_params']       = df_db['best_params'].astype(str)
    df_db['training_weeks']    = df_db['training_weeks'].astype(int)
    df_db['has_exog_features'] = df_db['has_exog_features'].astype(bool)

    print(f"   📋 准备写入 {len(df_db)} 行")
    print(f"   📋 列名及类型:\n{df_db.dtypes.to_string()}")
    print(f"   📋 数据预览:\n{df_db[['spu', 'run_date', 'forecast_target_date', 'spu_forecast_value']].head(3).to_string()}")

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
        pool_size=1,
        max_overflow=0,
        pool_recycle=300
    )
    try:
        with engine.begin() as conn:
            df_db.to_sql(
                'sales_forecast_history',
                con=conn,
                schema='finedatalink',
                if_exists='append',
                index=False,
                method='multi',
                chunksize=200
            )
        print(f"✅ 数据库写入成功! 目标表: finedatalink.sales_forecast_history，共写入 {len(df_db)} 条 SPU 级记录")

    except Exception as e:
        import traceback
        print("❌ 数据库写入失败，详细错误如下:")
        traceback.print_exc()
        backup_file = os.path.join('D:/华熠/output', f'backup_{datetime.date.today()}.csv')
        try:
            os.makedirs('D:/华熠/output', exist_ok=True)
            df_spu_level.to_csv(backup_file, index=False)
            print(f"   💾 已保存本地备份: {backup_file}")
        except Exception as backup_err:
            print(f"   ⚠️ 备份也失败了: {backup_err}")
    finally:
        engine.dispose()


def get_database_engine(db_url):
    """获取数据库引擎"""
    return create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
        pool_size=5,
        max_overflow=10,
        pool_recycle=300
    )


def test_database_connection(db_url):
    """测试数据库连接"""
    try:
        engine = get_database_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ 数据库连接成功")
        return True
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False
    finally:
        if 'engine' in locals():
            engine.dispose()