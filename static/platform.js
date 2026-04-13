const state = {
  selectionType: "all",
  resolvedSelection: null,
  currentRunId: null,
  currentRunStatus: "idle",
  pollingTimer: null,
  configs: [],
  resultLoadedFor: null,
  linuxOps: null,
  startingRun: false,
  stoppingRun: false,
  actionBusy: false,
  clientLogs: [],
  pendingLogTimer: null,
};

const selectionHelp = {
  all: "系统将自动解析当前可预测的全部 SPU。",
  manual: "支持换行、空格、逗号分隔的 SPU 清单。",
  sql: "请输入仅返回单列 spu 的 SELECT 语句，不符合将直接报错。",
};

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function nowIso() {
  return new Date().toISOString();
}

function setRunFeedback(message = "", tone = "info", visible = true) {
  const feedback = el("runFeedback");
  if (!feedback) return;
  feedback.className = `feedback feedback-${tone}${visible ? "" : " hidden"}`;
  feedback.textContent = message;
}

function setButtonBusy(buttonId, isBusy, busyLabel, idleLabel) {
  const button = el(buttonId);
  if (!button) return;
  button.disabled = isBusy;
  if (busyLabel && idleLabel) {
    button.textContent = isBusy ? busyLabel : idleLabel;
  }
}

function setStartButtonBusy(isBusy, label = "启动任务") {
  setButtonBusy("startJobBtn", isBusy, "提交中...", label);
}

function setStopBusy(isBusy) {
  setButtonBusy("stopJobBtn", isBusy, "停止中...", "停止任务");
}

function syncRunActionButtons() {
  const activeRun = ["queued", "running", "stopping"].includes(state.currentRunStatus);
  setStartButtonBusy(activeRun || state.startingRun || state.actionBusy);
  setButtonBusy("resolveSelectionBtn", activeRun || state.actionBusy);
  setStopBusy(state.stoppingRun);
}

function pushLocalLog(message, level = "info") {
  state.clientLogs.unshift({
    source: "前端",
    timestamp: nowIso(),
    message,
    level,
  });
  state.clientLogs = state.clientLogs.slice(0, 20);
  const container = el("logList");
  if (container) {
    container.innerHTML = renderLogs([...state.clientLogs], "当前暂无可展示日志");
    container.scrollTop = 0;
  }
}

function setActionBusy(isBusy, message = "", tone = "info") {
  state.actionBusy = isBusy;
  setButtonBusy("resolveSelectionBtn", isBusy);
  setButtonBusy("refreshLogsBtn", isBusy, "刷新中...", "刷新日志");
  setButtonBusy("refreshLinuxOpsBtn", isBusy, "刷新中...", "刷新运维状态");
  setButtonBusy("saveConfigBtn", isBusy, "保存中...", "保存配置");
  setButtonBusy("refreshConfigsBtn", isBusy, "刷新中...", "刷新列表");
  setButtonBusy("saveScheduleBtn", isBusy, "保存中...", "保存计划");
  if (message) {
    setRunFeedback(message, tone, true);
  }
}

function focusTaskStatus() {
  el("logList")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderLogs(logs, emptyText = "当前暂无可展示日志") {
  if (!logs.length) {
    return `<div class="log-entry">${escapeHtml(emptyText)}</div>`;
  }
  return logs
    .map((log) => {
      const css = log.level === "error" ? "log-entry error" : "log-entry";
      const prefix = [log.source, log.timestamp || log.created_at].filter(Boolean).join(" | ");
      const head = prefix ? `<span class="log-time">${escapeHtml(prefix)}</span>` : "";
      return `<div class="${css}">${head}${escapeHtml(log.message)}</div>`;
    })
    .join("");
}

function clearPendingClientLogs() {
  if (state.pendingLogTimer) {
    clearInterval(state.pendingLogTimer);
    state.pendingLogTimer = null;
  }
}

function startPendingClientLogs(message) {
  clearPendingClientLogs();
  pushLocalLog(message, "info");
  let heartbeat = 0;
  state.pendingLogTimer = setInterval(() => {
    heartbeat += 1;
    const suffix = heartbeat % 2 === 0
      ? "仍在处理，请稍候..."
      : "后端正在准备数据，请不要重复点击。";
    pushLocalLog(`${message} ${suffix}`, "info");
  }, 2500);
}

function setSelectionType(nextType) {
  state.selectionType = nextType || "all";
  document.querySelectorAll(".segment-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.selection === state.selectionType);
  });
  const manualInput = el("manualInput");
  if (manualInput) {
    manualInput.classList.toggle("hidden", state.selectionType !== "manual");
  }
  const sqlInput = el("sqlInput");
  if (sqlInput) {
    sqlInput.classList.toggle("hidden", state.selectionType !== "sql");
  }
  const help = el("selectionHelp");
  if (help) {
    help.textContent = selectionHelp[state.selectionType] || selectionHelp.all;
  }
}

function getSelectionPayload() {
  return { selection_type: "all" };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function setRunStatus(run) {
  const status = run?.status || "idle";
  state.currentRunStatus = status;
  const badge = el("runBadge");
  if (badge) {
    badge.className = `status status-${status}`;
    badge.textContent = status;
  }
  const heroRunStatus = el("heroRunStatus");
  if (heroRunStatus) {
    heroRunStatus.textContent = status;
  }
  syncRunActionButtons();
}

function updateRunSummary(run) {
  const summary = run?.summary || {};
  const failedSpus = summary.failed_spus ?? Math.max((run?.total_count || 0) - (run?.success_count || 0), 0);
  const items = [
    ["触发方式", run?.trigger_source || "-"],
    ["模式", run?.mode || "-"],
    ["平均 WMAPE", summary.average_wmape ? `${(summary.average_wmape * 100).toFixed(2)}%` : "-"],
    ["失败 SPU", failedSpus],
  ];
  const container = el("runSummary");
  if (container) {
    container.innerHTML = items
      .map(([label, value]) => `<div class="summary-item"><span class="muted">${label}</span><strong>${value}</strong></div>`)
      .join("");
  }
}

function updateResultKpis(bestRow) {
  const currentForecastValue = el("currentForecastValue");
  const bestModelValue = el("bestModelValue");
  const bestModelWmape = el("bestModelWmape");
  if (!bestRow) {
    if (currentForecastValue) currentForecastValue.textContent = "-";
    if (bestModelValue) bestModelValue.textContent = "-";
    if (bestModelWmape) bestModelWmape.textContent = "-";
    return;
  }
  const forecastText = `${bestRow.spu || "-"} 路 ${bestRow.spu_forecast_value ?? "-"}`;
  const wmapeText = bestRow.validation_wmape == null ? "-" : `${(bestRow.validation_wmape * 100).toFixed(2)}%`;
  if (currentForecastValue) currentForecastValue.textContent = forecastText;
  if (bestModelValue) bestModelValue.textContent = bestRow.winner_algo || "-";
  if (bestModelWmape) bestModelWmape.textContent = wmapeText;
}

function updateRunHeader(run) {
  if (!run) return;
  setRunStatus(run);
  const map = [
    ["progressValue", `${run.progress || 0}%`],
    ["processedValue", `${run.processed_count || 0} / ${run.total_count || 0}`],
    ["successValue", `${run.success_count || 0}`],
    ["currentSpuValue", run.current_spu || "-"],
    ["heroLastRun", run.id ? run.id.slice(0, 8) : "暂无"],
  ];
  map.forEach(([id, value]) => {
    const node = el(id);
    if (node) node.textContent = value;
  });
  const progressBar = el("progressBar");
  if (progressBar) progressBar.style.width = `${run.progress || 0}%`;
  updateRunSummary(run);
  const stopButton = el("stopJobBtn");
  if (stopButton) {
    const active = ["queued", "running", "stopping"].includes(run.status);
    stopButton.classList.toggle("hidden", !active);
  }
  if (run.status === "completed" && state.currentRunId && state.resultLoadedFor !== state.currentRunId) {
    loadResults().catch(handleError);
  }
}

async function resolveSelection() {
  const result = await api("/api/spu-selection/resolve", {
    method: "POST",
    body: JSON.stringify(getSelectionPayload()),
  });
  state.resolvedSelection = result;
  const count = el("selectionCount");
  if (count) count.textContent = `${result.count} 个`;
  const preview = el("selectionPreview");
  if (preview) {
    if (result.preview?.length) {
      preview.classList.remove("muted");
      preview.innerHTML = result.preview.map((spu) => `<span class="pill">${escapeHtml(spu)}</span>`).join("");
    } else {
      preview.classList.add("muted");
      preview.textContent = "暂无可解析 SPU";
    }
  }
  pushLocalLog(`SPU 范围已解析，共 ${result.count} 个。`, "info");
}

async function startRun() {
  if (state.actionBusy || state.startingRun) return;
  state.startingRun = true;
  setActionBusy(true, "任务已提交，正在进入执行队列...", "info");
  setStartButtonBusy(true);
  state.clientLogs = [];
  pushLocalLog("已点击启动，正在提交任务请求...", "info");
  focusTaskStatus();
  startPendingClientLogs("任务已提交，后端正在准备日志...");
  try {
    const result = await api("/api/forecast-jobs", {
      method: "POST",
      body: JSON.stringify({
        ...getSelectionPayload(),
        mode: el("modeSelect")?.value || "smart",
      }),
    });
    clearPendingClientLogs();
    state.currentRunId = result.run_id;
    updateRunHeader(result.run);
    pushLocalLog(`任务已创建，运行编号 ${result.run_id.slice(0, 8)}。`, "info");
    setRunFeedback("任务已启动，日志正在持续刷新。", "success", true);
    await loadRunStatus();
    startPolling();
  } finally {
    state.startingRun = false;
    setActionBusy(false);
    syncRunActionButtons();
  }
}

async function stopRun() {
  if (!state.currentRunId || state.stoppingRun) return;
  state.stoppingRun = true;
  setStopBusy(true);
  setRunFeedback("已发送停止请求，正在等待任务收尾。", "warning", true);
  pushLocalLog("已点击停止，正在发送停止请求...", "warning");
  startPendingClientLogs("正在发送停止请求，等待后端回传状态...");
  try {
    const run = await api(`/api/forecast-jobs/${state.currentRunId}/stop`, { method: "POST" });
    clearPendingClientLogs();
    pushLocalLog(`停止请求已送达，任务 ${run.id.slice(0, 8)} 正在收尾。`, "warning");
    updateRunHeader(run);
    await loadRunStatus();
  } finally {
    state.stoppingRun = false;
    syncRunActionButtons();
  }
}

async function loadRunStatus() {
  const runs = await api("/api/forecast-jobs?limit=1");
  if (!runs.length) {
    state.currentRunId = null;
    setRunStatus({ status: "idle" });
    ["progressValue", "processedValue", "successValue", "currentSpuValue"].forEach((id) => {
      const node = el(id);
      if (node) node.textContent = id === "progressValue" ? "0%" : id === "processedValue" ? "0 / 0" : "0";
      if (id === "currentSpuValue" && node) node.textContent = "-";
    });
    const progressBar = el("progressBar");
    if (progressBar) progressBar.style.width = "0%";
    const heroLastRun = el("heroLastRun");
    if (heroLastRun) heroLastRun.textContent = "暂无";
    const stopButton = el("stopJobBtn");
    if (stopButton) stopButton.classList.add("hidden");
    if (!state.startingRun && !state.stoppingRun) {
      setRunFeedback("", "info", false);
    }
    clearPendingClientLogs();
    syncRunActionButtons();
    await loadLinuxOps();
    await loadExecutionLogs();
    return;
  }
  const run = runs[0];
  state.currentRunId = run.id;
  updateRunHeader(run);
  if (run.status === "queued") {
    setRunFeedback("任务已进入队列，正在准备执行。", "info", true);
  } else if (run.status === "running") {
    setRunFeedback("任务正在执行，日志会持续刷新。", "success", true);
    clearPendingClientLogs();
  } else if (run.status === "stopping") {
    setRunFeedback("任务正在停止，请稍等。", "warning", true);
  } else if (run.status === "completed") {
    setRunFeedback("任务已完成，可以查看预测结果。", "success", true);
    clearPendingClientLogs();
    pushLocalLog(`任务 ${run.id.slice(0, 8)} 已完成。`, "info");
  } else if (run.status === "failed") {
    setRunFeedback("任务执行失败，请直接查看上方日志。", "warning", true);
    clearPendingClientLogs();
    pushLocalLog(`任务 ${run.id.slice(0, 8)} 执行失败。`, "error");
  }
  await Promise.all([loadRunSpus(), loadLinuxOps()]);
  await loadExecutionLogs();
  if (!["queued", "running", "stopping"].includes(run.status) && state.pollingTimer) {
    clearInterval(state.pollingTimer);
    state.pollingTimer = null;
  }
}

function startPolling() {
  if (state.pollingTimer) clearInterval(state.pollingTimer);
  loadRunStatus().catch(handleError);
  state.pollingTimer = setInterval(() => loadRunStatus().catch(handleError), 3000);
}

async function loadRunSpus() {
  if (!state.currentRunId) return;
  const spus = await api(`/api/forecast-jobs/${state.currentRunId}/spus`);
  const container = el("spuList");
  if (!container) return;
  if (!spus.length) {
    container.innerHTML = `<div class="stack-item muted">暂无 SPU 状态</div>`;
    return;
  }
  container.innerHTML = spus.map((item) => {
    const wmape = item.validation_wmape == null ? "-" : `${(item.validation_wmape * 100).toFixed(2)}%`;
    return `
      <article class="stack-item">
        <h4>${escapeHtml(item.spu)}</h4>
        <p>状态：${escapeHtml(item.status)}</p>
        <p>模型：${escapeHtml(item.winner_algo || "-")}</p>
        <p>WMAPE：${escapeHtml(wmape)}</p>
        <p>${escapeHtml(item.message || "")}</p>
      </article>
    `;
  }).join("");
}

async function loadLinuxOps() {
  const ops = await api("/api/linux-ops");
  state.linuxOps = ops;
  const summary = el("linuxOpsSummary");
  const sourceValue = el("linuxLogSourceValue");
  const nextRunValue = el("linuxNextRunValue");
  const serviceList = el("linuxServiceList");
  if (summary) summary.textContent = ops.captured_at ? `更新于 ${ops.captured_at}` : "已刷新";
  if (sourceValue) sourceValue.textContent = ops.logs?.source?.label || "服务日志";
  if (nextRunValue) nextRunValue.textContent = ops.next_schedule ? `${ops.next_schedule.config_name} · ${ops.next_schedule.next_run_text}` : "暂无启用的定期执行";
  const services = ops.services || [];
  if (!serviceList) return;
  if (!services.length) {
    serviceList.innerHTML = `<div class="stack-item muted">暂无系统服务状态</div>`;
    return;
  }
  serviceList.innerHTML = services.map((service) => {
    const stateText = service.status_text || `${service.active_state || "unknown"} / ${service.sub_state || "unknown"}`;
    const statusClass =
      service.active_state === "active"
        ? "status-running"
        : service.active_state === "inactive"
          ? "status-idle"
          : service.available === false
            ? "status-failed"
            : "status-stopped";
    return `
      <article class="stack-item">
        <div class="preview-topline">
          <strong>${escapeHtml(service.description || service.unit)}</strong>
          <span class="status ${statusClass}">${escapeHtml(service.active_state || "unknown")}</span>
        </div>
        <p>单位：${escapeHtml(service.unit || "-")}</p>
        <p>状态：${escapeHtml(stateText)}</p>
        <p>启用：${escapeHtml(service.unit_file_state || "-")}</p>
        <p>下次触发：${escapeHtml(service.next_elapse_realtime || "n/a")}</p>
        <p>日志：${escapeHtml(service.log_path || "n/a")}</p>
      </article>
    `;
  }).join("");
}

async function loadExecutionLogs() {
  const container = el("logList");
  if (!container) return;
  let logs = [];
  if (state.currentRunId) {
    try {
      logs = await api(`/api/forecast-jobs/${state.currentRunId}/logs?limit=200`);
    } catch (error) {
      console.warn("Failed to load run logs", error);
    }
  }

  const linuxLogSourceValue = el("linuxLogSourceValue");
  const linuxOpsSummary = el("linuxOpsSummary");
  const liveLogHint = el("liveLogHint");

  if (logs.length) {
    if (linuxLogSourceValue && state.currentRunId) {
      linuxLogSourceValue.textContent = `任务 ${state.currentRunId.slice(0, 8)}`;
    }
    if (linuxOpsSummary) linuxOpsSummary.textContent = "日志来源：当前运行任务";
    if (liveLogHint) liveLogHint.textContent = "当前显示：任务实时日志";
    const mergedLogs = [...state.clientLogs, ...logs].slice(0, 120);
    container.innerHTML = renderLogs(mergedLogs);
    container.scrollTop = container.scrollHeight;
    return;
  }

  if (state.linuxOps && linuxLogSourceValue) {
    linuxLogSourceValue.textContent = state.linuxOps.logs?.source?.label || "服务日志";
  }
  if (linuxOpsSummary) {
    const sourceParts = [];
    const source = state.linuxOps?.logs?.source || {};
    if (source.kind) sourceParts.push(source.kind);
    if (source.status) sourceParts.push(source.status);
    if (source.run_id) sourceParts.push(source.run_id.slice(0, 8));
    linuxOpsSummary.textContent = sourceParts.length ? `日志来源：${sourceParts.join(" / ")}` : "日志来源已更新";
  }
  if (liveLogHint) {
    liveLogHint.textContent = state.currentRunId ? "任务已提交，等待后端日志..." : "启动后会自动刷新";
  }

  const mergedLogs = state.clientLogs.length ? [...state.clientLogs] : [];
  const emptyText = state.currentRunId ? "任务已启动，正在等待后端日志..." : "当前暂无可展示日志";
  container.innerHTML = renderLogs(mergedLogs, emptyText);
  container.scrollTop = 0;
}

async function loadResults() {
  if (!state.currentRunId) {
    alert("暂无可查询的任务。");
    return;
  }
  const results = await api(`/api/forecast-results?run_id=${state.currentRunId}&limit=100`);
  const resultSummary = el("resultSummary");
  const resultsTableBody = el("resultsTableBody");
  if (!results.length) {
    if (resultSummary) resultSummary.textContent = "当前任务尚无结果数据。";
    if (resultsTableBody) resultsTableBody.innerHTML = `<tr><td colspan="6" class="muted">暂无结果</td></tr>`;
    updateResultKpis(null);
    return;
  }
  state.resultLoadedFor = state.currentRunId;
  const models = [...new Set(results.map((item) => item.winner_algo).filter(Boolean))];
  const validRows = results.filter((item) => item.validation_wmape != null);
  const bestRow = validRows.length
    ? [...validRows].sort((a, b) => a.validation_wmape - b.validation_wmape)[0]
    : results[0];
  updateResultKpis(bestRow);
  if (resultSummary) {
    if (models.length) {
      resultSummary.classList.remove("muted");
      resultSummary.innerHTML = models.map((model) => `<span class="pill">${escapeHtml(model)}</span>`).join("");
    } else {
      resultSummary.classList.add("muted");
      resultSummary.textContent = "暂无模型摘要";
    }
  }
  if (resultsTableBody) {
    resultsTableBody.innerHTML = results.map((row) => `
      <tr>
        <td>${escapeHtml(row.spu)}</td>
        <td>${escapeHtml(row.run_date)}</td>
        <td>${escapeHtml(row.forecast_target_date)}</td>
        <td>${escapeHtml(row.spu_forecast_value)}</td>
        <td>${escapeHtml(row.winner_algo || "-")}</td>
        <td>${row.validation_wmape == null ? "-" : `${(row.validation_wmape * 100).toFixed(2)}%`}</td>
      </tr>
    `).join("");
  }
}

async function saveConfig() {
  const payload = {
    name: el("configName")?.value.trim() || "",
    purpose: el("configPurpose")?.value.trim() || "",
    mode: el("modeSelect")?.value || "smart",
    ...getSelectionPayload(),
  };
  if (!payload.name) {
    throw new Error("请填写配置名称。");
  }
  setButtonBusy("saveConfigBtn", true, "保存中...", "保存配置");
  pushLocalLog("正在保存配置模板...", "info");
  try {
    await api("/api/forecast-configs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (el("configName")) el("configName").value = "";
    if (el("configPurpose")) el("configPurpose").value = "";
    await loadConfigs();
    await loadSchedules();
    pushLocalLog("配置模板保存成功。", "info");
  } finally {
    setButtonBusy("saveConfigBtn", false, "保存中...", "保存配置");
  }
}

async function loadConfigs() {
  const configs = await api("/api/forecast-configs");
  state.configs = configs;
  const list = el("configList");
  const select = el("scheduleConfigSelect");
  if (!configs.length) {
    if (list) list.innerHTML = `<div class="stack-item muted">暂无配置模板</div>`;
    if (select) select.innerHTML = `<option value="">暂无模板</option>`;
    return;
  }
  if (list) {
    list.innerHTML = configs.map((config) => `
      <article class="stack-item">
        <h4>${escapeHtml(config.name)}</h4>
        <p>${escapeHtml(config.purpose || "未填写用途")}</p>
        <p>模式：${escapeHtml(config.mode)} | 选择方式：${escapeHtml(config.selection_type)}</p>
        <div class="actions"><button class="btn btn-secondary" onclick="runConfig('${config.id}')">立即执行</button></div>
      </article>
    `).join("");
  }
  if (select) {
    select.innerHTML = configs.map((config) => `<option value="${config.id}">${escapeHtml(config.name)}</option>`).join("");
  }
}

async function runConfig(configId) {
  if (state.actionBusy || state.startingRun) return;
  state.startingRun = true;
  setActionBusy(true, "模板任务已提交，正在进入执行队列...", "info");
  setStartButtonBusy(true, "模板启动中...");
  state.clientLogs = [];
  pushLocalLog("已点击模板启动，正在提交任务请求...", "info");
  focusTaskStatus();
  startPendingClientLogs("模板任务已提交，后端正在准备日志...");
  try {
    const result = await api(`/api/forecast-configs/${configId}/run`, { method: "POST" });
    clearPendingClientLogs();
    state.currentRunId = result.run_id;
    updateRunHeader(result.run);
    pushLocalLog(`模板任务已创建，运行编号 ${result.run_id.slice(0, 8)}。`, "info");
    setRunFeedback("模板任务已启动，日志正在持续刷新。", "success", true);
    await loadRunStatus();
    startPolling();
  } finally {
    state.startingRun = false;
    setActionBusy(false);
    syncRunActionButtons();
  }
}

window.runConfig = runConfig;

async function saveSchedule() {
  const configId = el("scheduleConfigSelect")?.value;
  if (!configId) {
    throw new Error("请先选择模板。");
  }
  setButtonBusy("saveScheduleBtn", true, "保存中...", "保存计划");
  pushLocalLog("正在保存执行计划...", "info");
  try {
    await api("/api/forecast-schedules", {
      method: "POST",
      body: JSON.stringify({
        config_id: configId,
        weekday: Number(el("scheduleWeekday")?.value || 0),
        hour: Number(el("scheduleHour")?.value || 0),
        minute: Number(el("scheduleMinute")?.value || 0),
        timezone: "Asia/Shanghai",
        enabled: true,
      }),
    });
    await loadSchedules();
    pushLocalLog("执行计划保存成功。", "info");
  } finally {
    setButtonBusy("saveScheduleBtn", false, "保存中...", "保存计划");
  }
}

async function loadSchedules() {
  const schedules = await api("/api/forecast-schedules");
  const configMap = Object.fromEntries(state.configs.map((config) => [config.id, config]));
  const weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"];
  const list = el("scheduleList");
  if (!list) return;
  if (!schedules.length) {
    list.innerHTML = `<div class="stack-item muted">暂无执行计划</div>`;
    return;
  }
  list.innerHTML = schedules.map((schedule) => `
    <article class="stack-item">
      <h4>${escapeHtml(configMap[schedule.config_id]?.name || schedule.config_id)}</h4>
      <p>${weekdays[schedule.weekday]} ${String(schedule.hour).padStart(2, "0")}:${String(schedule.minute).padStart(2, "0")}</p>
      <p>时区：${escapeHtml(schedule.timezone)} | 状态：${schedule.enabled ? "启用" : "停用"}</p>
      <p>下次执行：${escapeHtml(schedule.next_run_text || "暂无")}</p>
    </article>
  `).join("");
}

function bindEvents() {
  document.querySelectorAll(".segment-btn").forEach((button) => {
    button.addEventListener("click", () => setSelectionType(button.dataset.selection));
  });
  el("resolveSelectionBtn")?.addEventListener("click", () => {
    if (state.actionBusy) return;
    setRunFeedback("正在解析 SPU 范围，请稍等。", "info", true);
    pushLocalLog("正在解析当前可预测 SPU 范围...", "info");
    resolveSelection().catch(handleError);
  });
  el("startJobBtn")?.addEventListener("click", () => startRun().catch(handleError));
  el("stopJobBtn")?.addEventListener("click", () => stopRun().catch(handleError));
  el("loadResultsBtn")?.addEventListener("click", () => {
    pushLocalLog("正在加载当前任务结果...", "info");
    loadResults().catch(handleError);
  });
  el("refreshLogsBtn")?.addEventListener("click", async () => {
    const button = el("refreshLogsBtn");
    try {
      setButtonBusy("refreshLogsBtn", true, "刷新中...", "刷新日志");
      setRunFeedback("正在刷新日志，请稍等。", "info", true);
      pushLocalLog("手动刷新日志区...", "info");
      await loadLinuxOps();
      await loadExecutionLogs();
    } catch (error) {
      handleError(error);
    } finally {
      if (button) button.disabled = false;
      setButtonBusy("refreshLogsBtn", false, "刷新中...", "刷新日志");
    }
  });
  el("refreshLinuxOpsBtn")?.addEventListener("click", async () => {
    try {
      setButtonBusy("refreshLinuxOpsBtn", true, "刷新中...", "刷新运维状态");
      pushLocalLog("正在刷新运维状态...", "info");
      await loadLinuxOps();
      await loadExecutionLogs();
    } catch (error) {
      handleError(error);
    } finally {
      setButtonBusy("refreshLinuxOpsBtn", false, "刷新中...", "刷新运维状态");
    }
  });
  el("saveConfigBtn")?.addEventListener("click", () => saveConfig().catch(handleError));
  el("refreshConfigsBtn")?.addEventListener("click", async () => {
    try {
      await loadConfigs();
      await loadSchedules();
    } catch (error) {
      handleError(error);
    }
  });
  el("saveScheduleBtn")?.addEventListener("click", () => saveSchedule().catch(handleError));
}

function handleError(error) {
  const actionLabel = state.stoppingRun ? "停止请求失败" : state.startingRun ? "启动任务失败" : "操作失败";
  state.startingRun = false;
  state.stoppingRun = false;
  state.actionBusy = false;
  syncRunActionButtons();
  setButtonBusy("resolveSelectionBtn", false);
  setButtonBusy("refreshLogsBtn", false, "刷新中...", "刷新日志");
  setButtonBusy("refreshLinuxOpsBtn", false, "刷新中...", "刷新运维状态");
  setButtonBusy("saveConfigBtn", false, "保存中...", "保存配置");
  setButtonBusy("refreshConfigsBtn", false, "刷新中...", "刷新列表");
  setButtonBusy("saveScheduleBtn", false, "保存中...", "保存计划");
  clearPendingClientLogs();
  pushLocalLog(`${actionLabel}: ${error.message || String(error)}`, "error");
  setRunFeedback(`${actionLabel}: ${error.message || String(error)}`, "warning", true);
}

async function bootstrap() {
  bindEvents();
  setSelectionType("all");
  setRunFeedback("", "info", false);
  updateResultKpis(null);
  syncRunActionButtons();
  await Promise.all([loadConfigs(), loadSchedules(), loadRunStatus(), loadLinuxOps()]);
  if (state.currentRunId) {
    startPolling();
  }
}

bootstrap().catch(handleError);
