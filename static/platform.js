const state = {
  selectionType: "all",
  resolvedSelection: null,
  currentRunId: null,
  pollingTimer: null,
  configs: [],
  resultLoadedFor: null,
};

const selectionHelp = {
  all: "系统将自动解析当前可预测的全部 SPU。",
  manual: "支持换行、空格、逗号分隔的 SPU 清单。",
  sql: "请输入仅返回单列 spu 的 SELECT 语句，不符合将直接报错。",
};

function el(id) {
  return document.getElementById(id);
}

function renderPills(containerId, values, emptyText = "暂无数据") {
  const container = el(containerId);
  if (!values || !values.length) {
    container.classList.add("muted");
    container.textContent = emptyText;
    return;
  }
  container.classList.remove("muted");
  container.innerHTML = values.map((value) => `<span class="pill">${value}</span>`).join("");
}

function setSelectionType(nextType) {
  state.selectionType = nextType;
  document.querySelectorAll(".segment-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.selection === nextType);
  });
  el("manualInput").classList.toggle("hidden", nextType !== "manual");
  el("sqlInput").classList.toggle("hidden", nextType !== "sql");
  el("selectionHelp").textContent = selectionHelp[nextType];
}

function getSelectionPayload() {
  return {
    selection_type: state.selectionType,
    manual_spus: el("manualInput").value,
    sql_query: el("sqlInput").value,
  };
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
  const badge = el("runBadge");
  badge.className = `status status-${status}`;
  badge.textContent = status;
  el("heroRunStatus").textContent = status;
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
  el("runSummary").innerHTML = items
    .map(([label, value]) => `<div class="summary-item"><span class="muted">${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function updateRunHeader(run) {
  if (!run) return;
  setRunStatus(run);
  el("progressValue").textContent = `${run.progress || 0}%`;
  el("processedValue").textContent = `${run.processed_count || 0} / ${run.total_count || 0}`;
  el("successValue").textContent = `${run.success_count || 0}`;
  el("currentSpuValue").textContent = run.current_spu || "-";
  el("progressBar").style.width = `${run.progress || 0}%`;
  el("heroLastRun").textContent = run.id ? run.id.slice(0, 8) : "暂无";
  updateRunSummary(run);
  const active = ["queued", "running", "stopping"].includes(run.status);
  el("stopJobBtn").classList.toggle("hidden", !active);
  if (run.status === "completed" && state.currentRunId && state.resultLoadedFor !== state.currentRunId) {
    loadResults().catch(handleError);
  }
}

function updateResultKpis(bestRow) {
  if (!bestRow) {
    el("currentForecastValue").textContent = "-";
    el("bestModelValue").textContent = "-";
    el("bestModelWmape").textContent = "-";
    return;
  }
  const forecastText = `${bestRow.spu || "-"} · ${bestRow.spu_forecast_value ?? "-"}`;
  const wmapeText =
    bestRow.validation_wmape == null ? "-" : `${(bestRow.validation_wmape * 100).toFixed(2)}%`;
  el("currentForecastValue").textContent = forecastText;
  el("bestModelValue").textContent = bestRow.winner_algo || "-";
  el("bestModelWmape").textContent = wmapeText;
}

async function resolveSelection() {
  const result = await api("/api/spu-selection/resolve", {
    method: "POST",
    body: JSON.stringify(getSelectionPayload()),
  });
  state.resolvedSelection = result;
  el("selectionCount").textContent = `${result.count} 个`;
  renderPills("selectionPreview", result.preview || [], "没有解析到 SPU");
}

async function startRun() {
  if (!state.resolvedSelection || state.resolvedSelection.selection_type !== state.selectionType) {
    await resolveSelection();
  }
  const result = await api("/api/forecast-jobs", {
    method: "POST",
    body: JSON.stringify({
      ...getSelectionPayload(),
      mode: el("modeSelect").value,
    }),
  });
  state.currentRunId = result.run_id;
  updateRunHeader(result.run);
  startPolling();
}

async function stopRun() {
  if (!state.currentRunId) return;
  const run = await api(`/api/forecast-jobs/${state.currentRunId}/stop`, { method: "POST" });
  updateRunHeader(run);
}

async function loadRunStatus() {
  const runs = await api("/api/forecast-jobs?limit=1");
  if (!runs.length) return;
  const run = runs[0];
  state.currentRunId = run.id;
  updateRunHeader(run);
  await Promise.all([loadRunSpus(), loadRunLogs()]);
  if (!["queued", "running", "stopping"].includes(run.status) && state.pollingTimer) {
    clearInterval(state.pollingTimer);
    state.pollingTimer = null;
  }
}

function startPolling() {
  if (state.pollingTimer) clearInterval(state.pollingTimer);
  loadRunStatus();
  state.pollingTimer = setInterval(loadRunStatus, 3000);
}

async function loadRunSpus() {
  if (!state.currentRunId) return;
  const spus = await api(`/api/forecast-jobs/${state.currentRunId}/spus`);
  const container = el("spuList");
  if (!spus.length) {
    container.innerHTML = `<div class="stack-item muted">暂无 SPU 状态</div>`;
    return;
  }
  container.innerHTML = spus.map((item) => {
    const wmape = item.validation_wmape == null ? "-" : `${(item.validation_wmape * 100).toFixed(2)}%`;
    return `
      <article class="stack-item">
        <h4>${item.spu}</h4>
        <p>状态：${item.status}</p>
        <p>模型：${item.winner_algo || "-"}</p>
        <p>WMAPE：${wmape}</p>
        <p>${item.message || ""}</p>
      </article>
    `;
  }).join("");
}

async function loadRunLogs() {
  if (!state.currentRunId) return;
  const logs = await api(`/api/forecast-jobs/${state.currentRunId}/logs?limit=200`);
  const container = el("logList");
  if (!logs.length) {
    container.innerHTML = `<div class="log-entry">等待日志输出...</div>`;
    return;
  }
  container.innerHTML = logs.map((log) => {
    const css = log.level === "error" ? "log-entry error" : "log-entry";
    return `<div class="${css}"><span class="log-time">${log.created_at}</span>${log.message}</div>`;
  }).join("");
  container.scrollTop = container.scrollHeight;
}

async function loadResults() {
  if (!state.currentRunId) {
    alert("暂无可查询的任务。");
    return;
  }
  const results = await api(`/api/forecast-results?run_id=${state.currentRunId}&limit=100`);
  if (!results.length) {
    el("resultSummary").textContent = "当前任务尚无结果数据。";
    el("resultsTableBody").innerHTML = `<tr><td colspan="6" class="muted">暂无结果</td></tr>`;
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
  renderPills("resultSummary", models, "暂无模型摘要");
  el("resultsTableBody").innerHTML = results.map((row) => `
    <tr>
      <td>${row.spu}</td>
      <td>${row.run_date}</td>
      <td>${row.forecast_target_date}</td>
      <td>${row.spu_forecast_value}</td>
      <td>${row.winner_algo || "-"}</td>
      <td>${row.validation_wmape == null ? "-" : `${(row.validation_wmape * 100).toFixed(2)}%`}</td>
    </tr>
  `).join("");
}

async function saveConfig() {
  const payload = {
    name: el("configName").value.trim(),
    purpose: el("configPurpose").value.trim(),
    mode: el("modeSelect").value,
    ...getSelectionPayload(),
  };
  if (!payload.name) {
    throw new Error("请填写配置名称。");
  }
  await api("/api/forecast-configs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  el("configName").value = "";
  el("configPurpose").value = "";
  await loadConfigs();
  await loadSchedules();
}

async function loadConfigs() {
  const configs = await api("/api/forecast-configs");
  state.configs = configs;
  const list = el("configList");
  const select = el("scheduleConfigSelect");
  if (!configs.length) {
    list.innerHTML = `<div class="stack-item muted">暂无配置模板</div>`;
    select.innerHTML = `<option value="">暂无模板</option>`;
    return;
  }
  list.innerHTML = configs.map((config) => `
    <article class="stack-item">
      <h4>${config.name}</h4>
      <p>${config.purpose || "未填写用途"}</p>
      <p>模式：${config.mode} | 选择方式：${config.selection_type}</p>
      <div class="actions"><button class="btn btn-secondary" onclick="runConfig('${config.id}')">立即执行</button></div>
    </article>
  `).join("");
  select.innerHTML = configs.map((config) => `<option value="${config.id}">${config.name}</option>`).join("");
}

async function runConfig(configId) {
  const result = await api(`/api/forecast-configs/${configId}/run`, { method: "POST" });
  state.currentRunId = result.run_id;
  updateRunHeader(result.run);
  startPolling();
}

window.runConfig = runConfig;

async function saveSchedule() {
  const configId = el("scheduleConfigSelect").value;
  if (!configId) {
    throw new Error("请先选择模板。");
  }
  await api("/api/forecast-schedules", {
    method: "POST",
    body: JSON.stringify({
      config_id: configId,
      weekday: Number(el("scheduleWeekday").value),
      hour: Number(el("scheduleHour").value),
      minute: Number(el("scheduleMinute").value),
      timezone: "Asia/Shanghai",
      enabled: true,
    }),
  });
  await loadSchedules();
}

async function loadSchedules() {
  const schedules = await api("/api/forecast-schedules");
  const configMap = Object.fromEntries(state.configs.map((config) => [config.id, config]));
  const weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"];
  const list = el("scheduleList");
  if (!schedules.length) {
    list.innerHTML = `<div class="stack-item muted">暂无执行计划</div>`;
    return;
  }
  list.innerHTML = schedules.map((schedule) => `
    <article class="stack-item">
      <h4>${configMap[schedule.config_id]?.name || schedule.config_id}</h4>
      <p>${weekdays[schedule.weekday]} ${String(schedule.hour).padStart(2, "0")}:${String(schedule.minute).padStart(2, "0")}</p>
      <p>时区：${schedule.timezone} | 状态：${schedule.enabled ? "启用" : "停用"}</p>
    </article>
  `).join("");
}

function bindEvents() {
  document.querySelectorAll(".segment-btn").forEach((button) => {
    button.addEventListener("click", () => setSelectionType(button.dataset.selection));
  });
  el("resolveSelectionBtn").addEventListener("click", () => resolveSelection().catch(handleError));
  el("startJobBtn").addEventListener("click", () => startRun().catch(handleError));
  el("stopJobBtn").addEventListener("click", () => stopRun().catch(handleError));
  el("loadResultsBtn").addEventListener("click", () => loadResults().catch(handleError));
  el("refreshLogsBtn").addEventListener("click", () => loadRunLogs().catch(handleError));
  el("saveConfigBtn").addEventListener("click", () => saveConfig().catch(handleError));
  el("refreshConfigsBtn").addEventListener("click", async () => {
    try {
      await loadConfigs();
      await loadSchedules();
    } catch (error) {
      handleError(error);
    }
  });
  el("saveScheduleBtn").addEventListener("click", () => saveSchedule().catch(handleError));
}

function handleError(error) {
  alert(error.message || String(error));
}

async function bootstrap() {
  bindEvents();
  setSelectionType("all");
  updateResultKpis(null);
  await Promise.all([loadConfigs(), loadSchedules(), loadRunStatus()]);
  if (state.currentRunId) {
    startPolling();
  }
}

bootstrap().catch(handleError);
