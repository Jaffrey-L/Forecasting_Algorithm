# Linux 上线运维说明

目标只覆盖当前第一版的两条验收标准：
1. Linux 定期后端执行
2. 固定服务可查看执行日志，运行时显示运行日志，支持临时执行，支持设置下次定期执行时间

## 服务单元

- `forecast-dashboard-v2.service`
- `forecast-weekly.service`
- `forecast-weekly.timer`

## 当前第一版的执行口径

- 正式执行面是 `forecast-dashboard-v2.service` 启动后的 API 服务内置调度器。
- 平台里保存的 `forecast_job_schedule` 会决定“下次定期执行时间”，并由 API 服务按计划触发任务。
- `forecast-weekly.timer` 仍保留，作为 Linux 运维侧的固定兜底任务，不是第一版里唯一的调度来源。
- 所以平台上“设置下次定期执行时间”当前指的是设置平台调度时间，不是自动改写 systemd timer。

## 启动方式

- API 服务：`systemctl enable --now forecast-dashboard-v2.service`
- Linux 兜底定时器：`systemctl enable --now forecast-weekly.timer`
- 定时任务本体由 `forecast-weekly.service` 执行 `scripts/run_forecast_job.py`

## 日志查看

- API 服务日志：`journalctl -u forecast-dashboard-v2.service -f`
- 定时任务日志：`journalctl -u forecast-weekly.service -f`
- 当前 systemd 单元同时落盘到：
  - `/var/log/forecast-dashboard-v2.log`
  - `/var/log/forecast-weekly.log`
- 平台日志区的规则：
  - 有运行中的任务时，优先展示任务运行日志
  - 没有运行中的任务时，回退展示服务日志

## 平台入口

- Linux 运维快照：`GET /api/linux-ops`
- 执行计划：`GET /api/forecast-schedules`
- 临时执行：平台“启动任务”按钮，触发 `POST /api/forecast-jobs`
- 保存定期执行时间：平台“保存计划”按钮，写入 `forecast_job_schedule`

## 下次执行时间

- 平台会根据 `forecast_job_schedule` 里的 `weekday / hour / minute / timezone` 计算下一次执行时间。
- 前端的“执行计划”区和“Linux 运维状态”区都会展示这个时间。
- 如果运维侧要让 `forecast-weekly.timer` 也保持同一时间，需要同步修改 timer 配置并执行 `systemctl daemon-reload`、`systemctl restart forecast-weekly.timer`。

## 验收口径

- Linux 上 `forecast-dashboard-v2.service` 正常运行，平台调度可用
- 如启用兜底任务，`forecast-weekly.timer` 正常启用
- 通过平台能看到服务状态
- 通过平台能看到当前运行日志或服务日志
- 通过平台能临时执行任务
- 通过平台能设置并看到下一次定期执行时间
