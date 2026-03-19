import os
import time
import json
import subprocess
import threading
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
static_dir = project_root

app = FastAPI(title="预测分析API", description="提供预测分析功能的API服务")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件服务
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 全局变量用于跟踪分析状态
analysis_status = {
    "status": "idle",  # idle, running, completed, failed
    "progress": 0,
    "processed_count": 0,
    "total_count": 0,
    "success_count": 0,
    "error_message": None
}

# 日志存储
analysis_logs = []

# 分析进程
analysis_process = None

# 根路径重定向到forecast_dashboard_v2.html
@app.get("/")
async def root():
    html_file_path = os.path.join(project_root, "forecast_dashboard_v2.html")
    print(f"[DEBUG] 正在加载HTML文件: {html_file_path}")
    print(f"[DEBUG] 文件是否存在: {os.path.exists(html_file_path)}")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="前端文件不存在")
    return FileResponse(html_file_path)

# 启动分析
@app.post("/api/start-analysis")
async def start_analysis(request: Request):
    global analysis_status, analysis_logs, analysis_process
    
    # 解析请求体
    data = await request.json()
    mode = data.get("mode", "fast")
    
    # 检查是否已经在运行
    if analysis_status["status"] == "running":
        return {"error": "分析已经在运行中"}
    
    # 重置状态
    analysis_status = {
        "status": "running",
        "progress": 0,
        "processed_count": 0,
        "total_count": 0,
        "success_count": 0,
        "error_message": None
    }
    analysis_logs = []
    
    # 启动分析进程
    def run_analysis():
        global analysis_status, analysis_logs, analysis_process
        try:
            # 构建命令
            command = [
                "python", "main.py",
                f"--mode={mode}"
            ]
            
            # 启动子进程
            analysis_process = subprocess.Popen(
                command,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # 读取输出
            for line in analysis_process.stdout:
                if line.strip():
                    analysis_logs.append(line.strip())
                    # 尝试解析进度信息
                    if "Progress:" in line:
                        try:
                            progress_str = line.split("Progress:")[1].strip().split("%")[0]
                            analysis_status["progress"] = int(progress_str)
                        except:
                            pass
                    if "Processed:" in line:
                        try:
                            processed_str = line.split("Processed:")[1].strip().split("/")[0]
                            total_str = line.split("/")[1].strip()
                            analysis_status["processed_count"] = int(processed_str)
                            analysis_status["total_count"] = int(total_str)
                        except:
                            pass
                    if "Success:" in line:
                        try:
                            success_str = line.split("Success:")[1].strip()
                            analysis_status["success_count"] = int(success_str)
                        except:
                            pass
            
            # 等待进程完成
            analysis_process.wait()
            
            if analysis_process.returncode == 0:
                analysis_status["status"] = "completed"
                analysis_status["progress"] = 100
            else:
                analysis_status["status"] = "failed"
                analysis_status["error_message"] = "分析执行失败"
                
        except Exception as e:
            analysis_status["status"] = "failed"
            analysis_status["error_message"] = str(e)
            analysis_logs.append(f"错误: {str(e)}")
    
    # 启动线程执行分析
    threading.Thread(target=run_analysis, daemon=True).start()
    
    return {"message": "分析已启动", "mode": mode}

# 停止分析
@app.post("/api/stop-analysis")
async def stop_analysis():
    global analysis_status, analysis_process
    
    if analysis_process and analysis_process.poll() is None:
        analysis_process.terminate()
        analysis_status["status"] = "idle"
        analysis_logs.append("用户手动停止分析")
        return {"message": "分析已停止"}
    else:
        return {"message": "没有正在运行的分析"}

# 获取分析状态
@app.get("/api/analysis-status")
async def get_analysis_status():
    return analysis_status

# 获取分析日志
@app.get("/api/analysis-logs")
async def get_analysis_logs():
    return {"logs": analysis_logs}

# 健康检查
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
