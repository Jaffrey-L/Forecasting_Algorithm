#!/usr/bin/env python3
"""
预测分析API - 测试版本
支持运行模式选择、实时日志和进度汇报
"""
import os
import sys
import json
import subprocess
import threading
import queue
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, Any, List
import uvicorn
import time

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

app = FastAPI(title="预测分析API(测试版)", description="支持实时日志和进度汇报的预测分析API")

# 配置静态文件服务
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 根路径重定向到新的前端页面
@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "forecast_dashboard_v2.html"))

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量用于存储分析状态
analysis_state = {
    "status": "idle",
    "progress": 0,
    "current_spu": "",
    "processed_count": 0,
    "total_count": 0,
    "logs": [],
    "completed_spus": [],
    "error": None,
    "result": None,
    "mode": "smart"
}

# 日志队列，用于实时流式传输
log_queue = queue.Queue()

# 后台任务：执行预测分析
def run_analysis_worker(mode='smart'):
    """执行预测分析工作进程"""
    global analysis_state
    
    try:
        analysis_state.update({
            "status": "running",
            "progress": 0,
            "current_spu": "",
            "processed_count": 0,
            "logs": [],
            "completed_spus": [],
            "error": None,
            "result": None,
            "mode": mode
        })
        
        db_url = os.getenv("SALES_FORECAST_DB_URL", "postgresql+psycopg2://postgres:vayiERty123@192.168.1.226:5432/finedatalink")
        
        # 构建命令 - 调用main_test.py
        worker_script = os.path.join(os.path.dirname(__file__), '../..', 'main_test.py')
        cmd = [sys.executable, worker_script, '--mode', mode, '--db-url', db_url]
        
        # 设置子进程环境变量，确保UTF-8编码
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        # 执行命令并捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并stderr到stdout
            text=True,
            encoding='utf-8',  # 使用UTF-8编码读取输出
            errors='replace',  # 遇到无法解码的字符时替换
            bufsize=1,  # 行缓冲
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
            env=env
        )
        
        # 读取输出并更新状态
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            
            # 解析进度
            if line.startswith('PROGRESS:'):
                progress = int(line.split(':')[1])
                analysis_state["progress"] = progress
            
            elif line.startswith('CURRENT_SPU:'):
                spu_name = line.split(':', 1)[1]
                analysis_state["current_spu"] = spu_name
            
            elif line.startswith('PROCESSED_COUNT:'):
                processed_count = int(line.split(':')[1])
                analysis_state["processed_count"] = processed_count
            
            elif line.startswith('TOTAL_COUNT:'):
                total_count = int(line.split(':')[1])
                analysis_state["total_count"] = total_count
            
            # 解析已完成的SPU
            elif line.startswith('COMPLETED_SPU:'):
                try:
                    completed_data = json.loads(line[14:])
                    analysis_state["completed_spus"].append(completed_data)
                except:
                    pass
            
            # 其他输出作为普通日志
            else:
                import datetime
                log_entry = {
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "type": "info",
                    "message": line
                }
                analysis_state["logs"].append(log_entry)
                log_queue.put(log_entry)
                # 限制日志数量
                if len(analysis_state["logs"]) > 1000:
                    analysis_state["logs"] = analysis_state["logs"][-500:]
        
        # 等待进程结束
        process.wait()
        
        if process.returncode != 0:
            analysis_state.update({
                "status": "failed",
                "error": f"分析进程异常退出，返回码: {process.returncode}"
            })
        else:
            analysis_state.update({"status": "completed", "progress": 100})
            
    except Exception as e:
        import traceback
        error_message = f"分析失败: {str(e)}"
        analysis_state.update({"status": "failed", "error": error_message})
        log_queue.put({"timestamp": "", "type": "error", "message": error_message})
        print(traceback.format_exc())

# API端点
@app.post("/api/start-analysis")
async def start_analysis(data: dict, background_tasks: BackgroundTasks):
    """启动分析任务"""
    global analysis_state
    
    if analysis_state["status"] == "running":
        raise HTTPException(status_code=400, detail="分析正在进行中")
    
    # 获取运行模式
    mode = data.get('mode', 'smart')
    if mode not in ['fast', 'smart', 'full']:
        raise HTTPException(status_code=400, detail="无效的运行模式，可选: fast, smart, full")
    
    # 重置状态
    analysis_state = {
        "status": "idle",
        "progress": 0,
        "current_spu": "",
        "processed_count": 0,
        "total_count": 0,
        "logs": [],
        "completed_spus": [],
        "error": None,
        "result": None,
        "mode": mode
    }
    
    # 清空日志队列
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except queue.Empty:
            break
    
    # 启动后台任务
    background_tasks.add_task(run_analysis_worker, mode)
    
    return {"message": "分析已启动", "status": "running", "mode": mode}

@app.get("/api/analysis-status")
async def get_analysis_status():
    """获取分析状态"""
    return analysis_state

@app.get("/api/analysis-logs")
async def get_analysis_logs(limit: int = 100):
    """获取最近的日志"""
    logs = analysis_state.get("logs", [])
    return {"logs": logs[-limit:]}

@app.get("/api/completed-spus")
async def get_completed_spus():
    """获取已完成的SPU列表"""
    return {"completed_spus": analysis_state.get("completed_spus", [])}

@app.get("/api/log-stream")
async def log_stream():
    """SSE流式传输日志"""
    async def event_generator():
        last_index = 0
        logs = analysis_state.get("logs", [])
        
        while True:
            logs = analysis_state.get("logs", [])
            
            # 发送新日志
            while last_index < len(logs):
                log_entry = logs[last_index]
                yield f"data: {json.dumps(log_entry, ensure_ascii=False)}\n\n"
                last_index += 1
            
            # 如果分析已完成，发送结束标记
            if analysis_state["status"] in ["completed", "failed"]:
                yield f"data: {json.dumps({'type': 'system', 'message': 'STREAM_END'}, ensure_ascii=False)}\n\n"
                break
                
            time.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "version": "2.0-test"}

@app.post("/api/stop-analysis")
async def stop_analysis():
    """停止分析任务"""
    global analysis_state
    
    if analysis_state["status"] != "running":
        raise HTTPException(status_code=400, detail="没有正在运行的分析任务")
    
    # 这里可以实现停止逻辑
    analysis_state["status"] = "stopped"
    analysis_state["error"] = "用户手动停止"
    
    return {"message": "分析已停止"}

if __name__ == "__main__":
    uvicorn.run("src.api.app_test:app", host="0.0.0.0", port=8000, reload=False)
