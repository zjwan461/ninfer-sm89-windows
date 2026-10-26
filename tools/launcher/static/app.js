/* NInfer Launcher 前端：无构建、无外部依赖（vanilla JS + EventSource）。 */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    schema: null,
    fields: [],
    fieldByKey: {},
    values: {},
    profiles: [],
    profileName: null,
    status: null,
    monitor: null,
    preview: null,
    logs: [],
    filter: "",
    autoScroll: true,
    paused: false,
    eventSource: null,
    busy: false,
  };

  /* ------------------------------------------------------------------ 工具 */

  const esc = (text) => String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));

  const fmtNum = (value, digits = 0) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    const n = Number(value);
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (Math.abs(n) >= 1e4) return (n / 1e3).toFixed(1) + "k";
    return n.toFixed(digits);
  };

  const fmtBytes = (bytes) => {
    const n = Number(bytes) || 0;
    if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(2) + " GiB";
    if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(1) + " MiB";
    if (n >= 1024) return (n / 1024).toFixed(0) + " KiB";
    return n + " B";
  };

  const fmtDuration = (seconds) => {
    if (seconds === null || seconds === undefined) return "—";
    const s = Math.max(0, Math.floor(seconds));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return (h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`);
  };

  const fmtRate = (tokens, seconds) => {
    const sec = Number(seconds) || 0;
    if (sec <= 0) return "—";
    return fmtNum(Number(tokens) / sec, 1) + " tok/s";
  };

  const clock = (ts) => {
    const d = new Date((Number(ts) || 0) * 1000);
    return d.toLocaleTimeString("zh-CN", { hour12: false });
  };

  function toast(message, kind = "") {
    const node = $("#toast");
    node.textContent = message;
    node.className = "toast " + kind;
    node.hidden = false;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { node.hidden = true; }, kind === "err" ? 8000 : 3500);
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let data = null;
    try { data = await response.json(); } catch { /* 非 JSON */ }
    if (!response.ok) {
      const detail = data && data.detail;
      const list = detail && detail.errors ? detail.errors : (typeof detail === "string" ? [detail] : ["请求失败"]);
      throw new Error(list.join("\n"));
    }
    return data;
  }

  const debounce = (fn, ms) => {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    };
  };

  /* ------------------------------------------------------------ 初始化 */

  // 角标显示的是「引擎构建」：dist 变体自己戳的标签（0.6.1-rtx4080s）或构建目录名。
  // 仓库根的 VERSION 是上游 RTX 3090 的标签，不能拿来做本 fork 的构建标识。
  function renderBuild(build) {
    const node = $("#version");
    if (!node) return;
    node.textContent = build ? "引擎 " + build : "";
    node.title = build ? "当前引擎构建：" + build : "";
    node.hidden = !build;
  }

  async function init() {
    try {
      const [schema, settings] = await Promise.all([api("/api/schema"), api("/api/settings")]);
      state.schema = schema;
      state.fields = schema.fields;
      state.fieldByKey = Object.fromEntries(schema.fields.map((f) => [f.key, f]));
      renderBuild(schema.build);
      state.values = { ...schema.defaults };
      if (settings.last_values && Object.keys(settings.last_values).length) {
        Object.assign(state.values, settings.last_values);
      }
      buildForm();
      buildModelsDatalist(schema.models_roots);
      bindEvents();
      refreshArtifactInfo();
      await refreshProfiles();
      await refreshModels();
      await refreshStatus();
      await renderPreview();
      connectLogs();
      setInterval(refreshStatus, 1000);
      setInterval(refreshMonitor, 2000);
    } catch (error) {
      toast("初始化失败：" + error.message, "err");
    }
  }

  /* -------------------------------------------------------------- 表单 */

  function buildForm() {
    const host = $("#form");
    host.innerHTML = "";
    for (const group of state.schema.groups) {
      const fields = state.fields.filter((f) => f.group === group.id);
      if (!fields.length) continue;
      const base = fields.filter((f) => !f.advanced);
      const advanced = fields.filter((f) => f.advanced);

      const section = document.createElement("section");
      section.className = "group";
      const title = document.createElement("h3");
      title.textContent = group.title;
      section.appendChild(title);
      for (const field of base) section.appendChild(buildRow(field));
      if (advanced.length) {
        const details = document.createElement("details");
        details.className = "advanced";
        const summary = document.createElement("summary");
        summary.textContent = `高级选项（${advanced.length}）`;
        details.appendChild(summary);
        for (const field of advanced) details.appendChild(buildRow(field));
        section.appendChild(details);
      }
      host.appendChild(section);
    }
    updateVisibility();
  }

  function buildRow(field) {
    const row = document.createElement("div");
    row.className = "field-row";
    row.dataset.key = field.key;

    const label = document.createElement("div");
    label.className = "label";
    label.innerHTML = `<div class="name">${esc(field.label)}</div>` +
      (field.help ? `<div class="hint">${esc(field.help)}</div>` : "");

    const control = document.createElement("div");
    control.className = "control";
    control.appendChild(buildControl(field));
    if (field.key === "artifact") {
      const hint = document.createElement("div");
      hint.className = "artifact-hint";
      hint.id = "artifact-hint";
      control.appendChild(hint);
    }

    row.appendChild(label);
    row.appendChild(control);
    return row;
  }

  function buildControl(field) {
    const id = "f-" + field.key;
    const value = state.values[field.key];
    let node;

    if (field.kind === "bool") {
      node = document.createElement("input");
      node.type = "checkbox";
      node.checked = value === true;
      node.addEventListener("change", () => onFieldChange(field.key));
    } else if (field.kind === "enum") {
      node = document.createElement("select");
      node.className = "select";
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = field.allow_empty ? "（默认 / 不启用）" : "（默认）";
      node.appendChild(empty);
      for (const choice of field.choices) {
        const option = document.createElement("option");
        option.value = choice;
        option.textContent = choice;
        node.appendChild(option);
      }
      node.value = value === null || value === undefined ? "" : String(value);
      node.addEventListener("change", () => onFieldChange(field.key));
    } else if (field.kind === "int" || field.kind === "float") {
      node = document.createElement("input");
      node.type = "number";
      node.className = "input num";
      if (field.kind === "float") { node.step = "any"; }
      if (field.min !== null && field.min !== undefined) node.min = field.min;
      if (field.max !== null && field.max !== undefined) node.max = field.max;
      node.placeholder = field.placeholder || "";
      node.value = value === null || value === undefined ? "" : String(value);
      node.addEventListener("input", () => onFieldChange(field.key));
    } else if (field.kind === "auto_int") {
      node = document.createElement("input");
      node.type = "text";
      node.className = "input num";
      node.setAttribute("list", "auto-list");
      node.placeholder = field.placeholder || "auto";
      node.value = value === null || value === undefined ? "" : String(value);
      node.addEventListener("input", () => onFieldChange(field.key));
    } else if (field.kind === "path") {
      node = document.createElement("input");
      node.type = "text";
      node.className = "input mono";
      node.placeholder = field.placeholder || "";
      if (field.key === "artifact") node.setAttribute("list", "models-list");
      node.value = value === null || value === undefined ? "" : String(value);
      node.addEventListener("input", () => onFieldChange(field.key));
    } else if (field.kind === "secret") {
      node = document.createElement("input");
      node.type = "password";
      node.className = "input";
      node.autocomplete = "off";
      node.value = value === null || value === undefined ? "" : String(value);
      node.addEventListener("input", () => onFieldChange(field.key));
    } else {
      node = document.createElement("input");
      node.type = "text";
      node.className = "input" + (field.kind === "raw" ? " mono" : "");
      node.placeholder = field.placeholder || "";
      node.value = value === null || value === undefined ? "" : String(value);
      node.addEventListener("input", () => onFieldChange(field.key));
    }

    node.id = id;
    return node;
  }

  function readControl(field) {
    const node = document.getElementById("f-" + field.key);
    if (!node) return state.values[field.key];
    if (field.kind === "bool") return node.checked;
    const raw = String(node.value ?? "").trim();
    if (raw === "") return null;
    if (field.kind === "int") {
      const n = Number(raw);
      return Number.isFinite(n) ? Math.trunc(n) : raw;
    }
    if (field.kind === "float") {
      const n = Number(raw);
      return Number.isFinite(n) ? n : raw;
    }
    if (field.kind === "auto_int") return raw.toLowerCase() === "auto" ? "auto" : raw;
    return raw;
  }

  function onFieldChange(key) {
    const field = state.fieldByKey[key];
    if (!field) return;
    state.values[key] = readControl(field);
    updateVisibility();
    if (key === "artifact") scheduleArtifactInfo();
    schedulePreview();
  }

  function updateVisibility() {
    for (const field of state.fields) {
      const rule = field.show_if;
      const row = document.querySelector(`.field-row[data-key="${field.key}"]`);
      if (!row) continue;
      if (!rule) { row.classList.remove("hidden"); continue; }
      const value = state.values[rule.key];
      let visible = true;
      if (Array.isArray(rule.values)) visible = rule.values.includes(value);
      else if (rule.nonempty) visible = value !== null && value !== undefined && String(value).trim() !== "";
      row.classList.toggle("hidden", !visible);
    }
  }

  function applyValues(values) {
    for (const field of state.fields) {
      if (values[field.key] === undefined) continue;
      state.values[field.key] = values[field.key];
      const node = document.getElementById("f-" + field.key);
      if (!node) continue;
      const value = values[field.key];
      if (field.kind === "bool") node.checked = value === true;
      else node.value = value === null || value === undefined ? "" : String(value);
    }
    updateVisibility();
    scheduleArtifactInfo();
    schedulePreview();
  }

  /* --------------------------------------------------- artifact 能力探测 */

  // 引擎按组件名取权重：--spec 要 mtp/dflash/dflash2、--vision 要 vision、
  // --lm-head-draft 要 proposal。缺失时在加载阶段就直接 FATAL，这里提前提示并禁用。
  const scheduleArtifactInfo = debounce(() => { refreshArtifactInfo(); }, 350);

  async function refreshArtifactInfo() {
    const node = $("#artifact-hint");
    if (!node) return;
    const path = state.values.artifact;
    if (!path) {
      node.textContent = "";
      node.className = "artifact-hint";
      applyArtifactCapabilities(null);
      return;
    }
    try {
      const info = await api("/api/artifact?path=" + encodeURIComponent(path));
      if (!info.ok) {
        node.className = "artifact-hint warn";
        node.textContent = "无法读取 artifact：" + (info.error || "未知错误");
        applyArtifactCapabilities(null);
        return;
      }
      const parts = ["组件：" + (info.components.join(", ") || "无")];
      if (info.architecture) parts.push("架构 " + info.architecture);
      if (info.max_context) parts.push("上下文上限 " + info.max_context);
      if (info.name) parts.push(info.name);
      node.className = "artifact-hint";
      node.textContent = parts.join(" · ");
      applyArtifactCapabilities(info);
    } catch (error) {
      node.className = "artifact-hint warn";
      node.textContent = "读取 artifact 失败：" + error.message;
      applyArtifactCapabilities(null);
    }
  }

  function applyArtifactCapabilities(info) {
    const spec = document.getElementById("f-spec");
    if (spec) {
      const allowed = info && info.ok ? new Set(info.speculative_backends) : null;
      for (const option of spec.options) {
        if (!option.value) continue;
        const supported = !allowed || allowed.has(option.value);
        option.disabled = !supported;
        option.title = supported ? "" : "该 artifact 不含此组件";
      }
    }
    for (const [key, supported] of [["vision", info.components.includes("vision")], ["lm_head_draft", info.proposal]]) {
      const node = document.getElementById("f-" + key);
      if (!node || !info || !info.ok) {
        if (node) { node.disabled = false; node.title = ""; }
        continue;
      }
      node.disabled = !supported;
      node.title = supported ? "" : "该 artifact 不含 " + (key === "vision" ? "vision" : "proposal") + " 组件";
    }
  }

  const schedulePreview = debounce(() => { renderPreview(); }, 280);

  async function renderPreview() {
    try {
      const result = await api("/api/preview", {
        method: "POST",
        body: JSON.stringify({ values: state.values }),
      });
      state.preview = result;
      $("#cmdline").textContent = result.display || "—";

      const box = $("#validation");
      const notes = [];
      for (const error of result.errors) notes.push(`<div class="err">${esc(error)}</div>`);
      for (const warning of result.warnings) notes.push(`<div class="warn">${esc(warning)}</div>`);
      if (!notes.length) notes.push('<div class="ok">参数校验通过</div>');
      box.innerHTML = notes.join("");

      const running = state.status && state.status.argv && state.status.argv.length;
      const dirty = running && JSON.stringify(state.status.argv) !== JSON.stringify(result.argv);
      $("#dirty-hint").textContent = dirty ? "配置与正在运行的实例不同，重启后生效" : "";
    } catch (error) {
      $("#validation").innerHTML = `<div class="err">${esc(error.message)}</div>`;
    }
  }

  function buildModelsDatalist(roots) {
    const list = $("#models-list");
    list.innerHTML = "";
  }

  async function refreshModels() {
    try {
      const result = await api("/api/models");
      const list = $("#models-list");
      list.innerHTML = "";
      for (const model of result.models) {
        const option = document.createElement("option");
        option.value = model.path;
        option.label = `${model.name} · ${(model.size / 1024 ** 3).toFixed(1)} GiB`;
        list.appendChild(option);
      }
    } catch { /* 忽略：模型扫描是辅助功能 */ }
  }

  /* -------------------------------------------------------------- 状态 */

  const STATE_LABEL = {
    stopped: ["已停止", "pill-idle"],
    starting: ["加载中…", "pill-starting"],
    ready: ["运行中", "pill-ready"],
    crashed: ["已崩溃", "pill-crashed"],
    external: ["外部实例", "pill-external"],
  };

  async function refreshStatus() {
    try {
      const status = await api("/api/status");
      state.status = status;
      renderStatus(status);
    } catch { /* launcher 重启期间忽略 */ }
  }

  function renderStatus(status) {
    const [label, klass] = STATE_LABEL[status.state] || [status.state, "pill-idle"];
    const pill = $("#status-pill");
    pill.textContent = label;
    pill.className = "pill " + klass;
    renderBuild(status.build);

    const detail = [];
    if (status.pid) detail.push(`PID ${status.pid}`);
    if (status.external_pid && status.state === "external") detail.push(`占用 PID ${status.external_pid}`);
    if (status.state === "ready") detail.push(`就绪 ${fmtDuration(status.uptime_s)}`);
    if (status.state === "starting") detail.push(`已等待 ${fmtDuration(status.uptime_s)}（加载权重）`);
    if (status.exit_code !== null && status.exit_code !== undefined && status.state === "crashed") {
      detail.push(`退出码 ${status.exit_code}`);
    }
    $("#status-detail").textContent = detail.join(" · ");

    const running = status.state === "starting" || status.state === "ready";
    setDisabled("#btn-start", running || status.state === "external");
    setDisabled("#btn-start-2", running || status.state === "external");
    setDisabled("#btn-stop", !running);
    setDisabled("#btn-stop-2", !running);
    $("#btn-kill-external").hidden = !(status.state === "external" && status.external_pid);

    // 总览卡片
    $("#ov-state").textContent = label;
    $("#ov-uptime").textContent = status.started_at
      ? `启动于 ${clock(status.started_at)} · ${fmtDuration(status.uptime_s)}`
      : (status.state === "external" ? "由外部进程启动" : "—");

    const argv = status.argv || [];
    const artifact = argv[1] || "";
    $("#ov-artifact").textContent = artifact || "—";
    const modelId = findFlag(argv, "--model-id");
    $("#ov-model").textContent = modelId || (artifact ? artifact.split(/[\\/]/).pop() : "—");

    const gpu = status.gpu || {};
    if (gpu.ok) {
      const used = gpu.used_mib / 1024, total = gpu.total_mib / 1024;
      $("#ov-vram").textContent = `${used.toFixed(1)} / ${total.toFixed(0)} GiB`;
      $("#ov-vram-sub").textContent = `空闲 ${(gpu.free_mib / 1024).toFixed(1)} GiB` +
        (gpu.util_percent !== null && gpu.util_percent !== undefined ? ` · 利用率 ${gpu.util_percent}%` : "") +
        (gpu.temp_c !== null && gpu.temp_c !== undefined ? ` · ${gpu.temp_c}°C` : "");
      const pct = total > 0 ? Math.min(100, (used / total) * 100) : 0;
      $("#ov-vram-bar").style.width = pct.toFixed(1) + "%";
      $("#ov-vram-bar").style.background = pct > 92
        ? "linear-gradient(90deg,#f2616b,#e8b04b)"
        : "linear-gradient(90deg,var(--accent),var(--violet))";
    } else {
      $("#ov-vram").textContent = "—";
      $("#ov-vram-sub").textContent = gpu.error || "无数据";
      $("#ov-vram-bar").style.width = "0%";
    }

    const host = status.host || "127.0.0.1";
    const port = status.port || 8080;
    const base = `http://${host}:${port}`;
    $("#ov-endpoint").textContent = base;
    for (const link of $$("#ov-links a")) {
      const kind = link.dataset.link;
      link.href = kind === "base" ? base + "/v1" : base + (kind === "monitor" ? "/monitor" : kind === "metrics" ? "/metrics" : "/health");
    }

    $("#ov-cmd").textContent = argv.length ? status.argv.join(" ") : "（尚未启动）";
  }

  function setDisabled(selector, disabled) {
    const node = $(selector);
    if (node) node.disabled = !!disabled;
  }

  function findFlag(argv, flag) {
    const index = argv.indexOf(flag);
    return index >= 0 && index + 1 < argv.length ? argv[index + 1] : null;
  }

  /* -------------------------------------------------------------- 监控 */

  async function refreshMonitor() {
    const status = state.status;
    if (!status || (status.state !== "ready" && status.state !== "external")) {
      state.monitor = null;
      renderMonitor(null);
      return;
    }
    try {
      const data = await api("/api/monitor");
      if (data && data.ok === false) { renderMonitor(null); return; }
      state.monitor = data;
      renderMonitor(data);
    } catch { /* 忽略瞬时失败 */ }
  }

  function renderMonitor(data) {
    if (!data) {
      for (const id of ["ov-prefill", "ov-decode", "ov-kv", "ov-sched", "ov-requests", "ov-cache", "ov-draft", "ov-ctx"]) {
        const node = document.getElementById(id);
        if (node) node.textContent = "—";
      }
      $("#ov-slots").innerHTML = "";
      $("#ov-recent").innerHTML = "";
      $("#ov-monitor-stamp").textContent = "";
      return;
    }
    const totals = data.totals || {};
    const kv = data.kv || {};
    const scheduler = data.scheduler || {};

    $("#ov-prefill").textContent = fmtRate(totals.prefill_tokens, totals.prefill_seconds);
    $("#ov-decode").textContent = fmtRate(totals.decode_tokens, totals.decode_seconds);
    $("#ov-kv").textContent = kv.pages
      ? `${fmtNum(kv.occupied_pages)} / ${fmtNum(kv.pages)} 页`
      : "—";
    $("#ov-sched").textContent = `${scheduler.in_flight || 0} / ${data.lanes || "?"} 并发` +
      (scheduler.waiting ? ` · 排队 ${scheduler.waiting}` : "");
    $("#ov-requests").textContent = fmtNum(totals.requests);
    $("#ov-cache").textContent = fmtNum(totals.cached_prompt_tokens) + " tok";
    const drafted = Number(totals.drafted) || 0, accepted = Number(totals.accepted) || 0;
    $("#ov-draft").textContent = drafted > 0 ? ((accepted / drafted) * 100).toFixed(1) + "%" : "—";
    $("#ov-ctx").textContent = fmtNum(data.max_context) + " tok";
    $("#ov-monitor-stamp").textContent = "每 2 秒刷新 · " + (data.model || "");

    const slots = data.slots || [];
    $("#ov-slots").innerHTML = slots.length
      ? "<table class=\"tbl\"><tr><th>槽位</th><th>处理中</th><th>保留</th><th>prompt</th><th>缓存</th><th>检查点</th></tr>" +
        slots.map((s) => `<tr><td>${s.id}</td><td>${s.processing ? "是" : ""}</td><td>${s.retained ? "是" : ""}</td>` +
          `<td>${fmtNum(s.prompt_tokens)}</td><td>${fmtNum(s.cached_tokens)}</td><td>${s.checkpoints}</td></tr>`).join("") +
        "</table>"
      : "";

    const recent = (data.recent || []).slice(0, 8);
    $("#ov-recent").innerHTML = recent.length
      ? "<table class=\"tbl\"><tr><th>#</th><th>prompt</th><th>复用</th><th>输出</th><th>TTFT</th><th>decode</th><th>草稿</th><th>结束</th></tr>" +
        recent.map((r) => `<tr><td>${r.sequence}</td><td>${fmtNum(r.prompt_tokens)}</td>` +
          `<td>${fmtNum(r.cached_tokens)}</td><td>${fmtNum(r.completion_tokens)}</td>` +
          `<td>${(Number(r.ttft_seconds) * 1000).toFixed(0)} ms</td>` +
          `<td>${fmtNum(Number(r.completion_tokens) / Math.max(Number(r.decode_seconds), 1e-9), 1)} tok/s</td>` +
          `<td>${r.drafted ? `${r.accepted}/${r.drafted}` : "—"}</td><td>${esc(r.finish_reason || "")}</td></tr>`).join("") +
        "</table>"
      : "";
  }

  /* -------------------------------------------------------------- 日志 */

  function connectLogs() {
    if (state.eventSource) state.eventSource.close();
    const source = new EventSource("/api/logs?history=400");
    state.eventSource = source;
    source.onmessage = (event) => {
      let item;
      try { item = JSON.parse(event.data); } catch { return; }
      if (!item || typeof item.text !== "string") return;
      state.logs.push(item);
      if (state.logs.length > 3000) state.logs.splice(0, state.logs.length - 3000);
      if (!state.paused) appendLogLine(item);
      $("#log-count").textContent = `${state.logs.length} 行`;
    };
    source.onerror = () => { /* EventSource 自动重连 */ };
  }

  function logLineHtml(item) {
    const cls = item.stream === "launcher" ? "line launcher" : "line";
    const prefix = item.stream === "launcher" ? "● " : "  ";
    return `<span class="${cls}"><span class="ts">[${clock(item.ts)}]</span> ${prefix}${esc(item.text)}</span>`;
  }

  function matchesFilter(item) {
    if (!state.filter) return true;
    return item.text.toLowerCase().includes(state.filter);
  }

  function appendLogLine(item) {
    if (!matchesFilter(item)) return;
    const view = $("#log-view");
    view.insertAdjacentHTML("beforeend", logLineHtml(item) + "\n");
    if (state.autoScroll) view.scrollTop = view.scrollHeight;
    const mini = $("#ov-log");
    mini.insertAdjacentHTML("beforeend", logLineHtml(item) + "\n");
    if (mini.textContent.split("\n").length > 60) {
      mini.innerHTML = mini.innerHTML.split("\n").slice(-40).join("\n");
    }
    mini.scrollTop = mini.scrollHeight;
  }

  function rerenderLogs() {
    const view = $("#log-view");
    view.innerHTML = state.logs.filter(matchesFilter).map(logLineHtml).join("\n") + "\n";
    if (state.autoScroll) view.scrollTop = view.scrollHeight;
    const mini = $("#ov-log");
    const tail = state.logs.filter(matchesFilter).slice(-40);
    mini.innerHTML = tail.map(logLineHtml).join("\n") + "\n";
    mini.scrollTop = mini.scrollHeight;
  }

  /* -------------------------------------------------------------- 档案 */

  async function refreshProfiles() {
    try {
      const result = await api("/api/profiles");
      state.profiles = result.profiles;
      const select = $("#profile-select");
      select.innerHTML = '<option value="">（未命名配置）</option>';
      for (const profile of state.profiles) {
        const option = document.createElement("option");
        option.value = profile.name;
        option.textContent = profile.builtin ? `${profile.name}` : profile.name;
        select.appendChild(option);
      }
      if (result.last) {
        select.value = result.last;
        state.profileName = result.last;
      }
      $("#profile-status").textContent = `${state.profiles.length} 个档案`;
    } catch (error) {
      $("#profile-status").textContent = "读取档案失败：" + error.message;
    }
  }

  function loadProfile() {
    const name = $("#profile-select").value;
    if (!name) {
      state.profileName = null;
      $("#profile-status").textContent = "未载入档案";
      return;
    }
    const profile = state.profiles.find((item) => item.name === name);
    if (!profile) return;
    state.profileName = name;
    applyValues(profile.values);
    $("#profile-status").textContent = `已载入：${name}`;
    toast(`已载入档案「${name}」`);
  }

  async function saveProfile() {
    const suggestion = state.profileName || "我的配置";
    const name = window.prompt("档案名（同名覆盖）", suggestion);
    if (!name || !name.trim()) return;
    try {
      await api("/api/profiles", {
        method: "POST",
        body: JSON.stringify({ name: name.trim(), values: state.values }),
      });
      state.profileName = name.trim();
      await refreshProfiles();
      $("#profile-select").value = state.profileName;
      toast(`已保存档案「${state.profileName}」`, "ok");
    } catch (error) {
      toast("保存失败：" + error.message, "err");
    }
  }

  async function deleteProfile() {
    const name = $("#profile-select").value;
    if (!name) return;
    if (!window.confirm(`删除档案「${name}」？`)) return;
    try {
      await api("/api/profiles?name=" + encodeURIComponent(name), { method: "DELETE" });
      if (state.profileName === name) state.profileName = null;
      await refreshProfiles();
      toast(`已删除「${name}」`, "ok");
    } catch (error) {
      toast("删除失败：" + error.message, "err");
    }
  }

  function exportBat() {
    const name = $("#profile-select").value;
    if (!name) { toast("先选择一个档案", "err"); return; }
    window.location.href = "/api/profiles/bat?name=" + encodeURIComponent(name);
  }

  async function restorePresets() {
    if (!window.confirm("重新导入内置预设？同名档案会被覆盖。")) return;
    try {
      const result = await api("/api/profiles/presets?overwrite=true", { method: "POST" });
      await refreshProfiles();
      toast(`已恢复 ${result.added.length} 个预设`, "ok");
    } catch (error) {
      toast("恢复失败：" + error.message, "err");
    }
  }

  /* -------------------------------------------------------------- 启停 */

  async function startServe() {
    if (state.busy) return;
    state.busy = true;
    setDisabled("#btn-start", true);
    setDisabled("#btn-start-2", true);
    try {
      await api("/api/start", { method: "POST", body: JSON.stringify({ values: state.values }) });
      toast("serve 启动中，正在加载权重…", "ok");
      switchTab("overview");
      await refreshStatus();
    } catch (error) {
      toast(error.message, "err");
      await renderPreview();
    } finally {
      state.busy = false;
      await refreshStatus();
    }
  }

  async function stopServe() {
    if (state.busy) return;
    state.busy = true;
    try {
      await api("/api/stop", { method: "POST" });
      toast("serve 已停止", "ok");
    } catch (error) {
      toast(error.message, "err");
    } finally {
      state.busy = false;
      await refreshStatus();
    }
  }

  async function killExternal() {
    const pid = state.status && state.status.external_pid;
    if (!pid) return;
    if (!window.confirm(`结束占用端口的进程 PID ${pid}？`)) return;
    try {
      const result = await api("/api/kill-external", {
        method: "POST",
        body: JSON.stringify({ pid }),
      });
      toast(result.message, "ok");
    } catch (error) {
      toast(error.message, "err");
    }
    await refreshStatus();
  }

  /* -------------------------------------------------------------- 测试 */

  let testRunning = false;

  async function sendTest() {
    if (testRunning) return;
    const prompt = $("#test-prompt").value.trim();
    if (!prompt) { toast("先写一句提示词", "err"); return; }
    const maxTokens = Number($("#test-max-tokens").value) || 256;
    const temperature = Number($("#test-temperature").value);

    testRunning = true;
    setDisabled("#btn-test-send", true);
    $("#test-status").textContent = "请求中…";
    $("#test-answer").textContent = "";
    for (const id of ["tt-prefill", "tt-decode", "tt-cache", "tt-draft"]) {
      document.getElementById(id).textContent = "—";
    }

    let answer = "", reasoning = "";
    const renderAnswer = () => {
      $("#test-answer").textContent = reasoning
        ? `⟨思考⟩\n${reasoning}\n\n⟨回答⟩\n${answer}`
        : answer;
      $("#test-answer").scrollTop = $("#test-answer").scrollHeight;
    };

    try {
      const response = await fetch("/api/test/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_tokens: maxTokens, temperature }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        const list = detail.detail && detail.detail.errors ? detail.detail.errors : ["请求失败"];
        throw new Error(list.join("\n"));
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          for (const line of block.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (!payload || payload === "[DONE]") continue;
            let object;
            try { object = JSON.parse(payload); } catch { continue; }
            if (object.error) { throw new Error(String(object.error)); }
            const choice = object.choices && object.choices[0];
            if (choice && choice.delta) {
              if (choice.delta.reasoning_content) { reasoning += choice.delta.reasoning_content; renderAnswer(); }
              if (choice.delta.content) { answer += choice.delta.content; renderAnswer(); }
            }
            if (object.timings) renderTimings(object.timings);
          }
        }
      }
      $("#test-status").textContent = "完成";
    } catch (error) {
      $("#test-status").textContent = "失败";
      toast(error.message, "err");
    } finally {
      testRunning = false;
      setDisabled("#btn-test-send", false);
    }
  }

  function renderTimings(timings) {
    if (!timings) return;
    $("#tt-prefill").textContent = timings.prompt_per_second
      ? fmtNum(timings.prompt_per_second, 1) + " tok/s" : "—";
    $("#tt-decode").textContent = timings.predicted_per_second
      ? fmtNum(timings.predicted_per_second, 1) + " tok/s" : "—";
    $("#tt-cache").textContent = timings.cache_n !== undefined
      ? fmtNum(timings.cache_n) + " tok" : "—";
    $("#tt-draft").textContent = timings.draft_n
      ? `${timings.draft_n_accepted || 0}/${timings.draft_n}` : "—";
  }

  /* -------------------------------------------------------------- 绑定 */

  function switchTab(name) {
    for (const button of $$(".tab-btn")) button.classList.toggle("active", button.dataset.tab === name);
    for (const tab of $$(".tab")) tab.classList.toggle("active", tab.id === "tab-" + name);
    if (name === "config") renderPreview();
    if (name === "logs" && state.autoScroll) {
      const view = $("#log-view");
      view.scrollTop = view.scrollHeight;
    }
  }

  function bindEvents() {
    $("#tabs").addEventListener("click", (event) => {
      const button = event.target.closest(".tab-btn");
      if (button) switchTab(button.dataset.tab);
    });
    for (const link of $$("[data-goto]")) {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        switchTab(link.dataset.goto);
      });
    }

    $("#btn-start").addEventListener("click", startServe);
    $("#btn-start-2").addEventListener("click", startServe);
    $("#btn-stop").addEventListener("click", stopServe);
    $("#btn-stop-2").addEventListener("click", stopServe);
    $("#btn-kill-external").addEventListener("click", killExternal);

    $("#btn-copy-cmd").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText($("#ov-cmd").textContent);
        toast("命令行已复制", "ok");
      } catch { toast("复制失败", "err"); }
    });

    $("#btn-profile-load").addEventListener("click", loadProfile);
    $("#profile-select").addEventListener("change", loadProfile);
    $("#btn-profile-save").addEventListener("click", saveProfile);
    $("#btn-profile-delete").addEventListener("click", deleteProfile);
    $("#btn-profile-bat").addEventListener("click", exportBat);
    $("#btn-profile-presets").addEventListener("click", restorePresets);

    $("#log-filter").addEventListener("input", (event) => {
      state.filter = event.target.value.trim().toLowerCase();
      rerenderLogs();
    });
    $("#log-autoscroll").addEventListener("change", (event) => {
      state.autoScroll = event.target.checked;
      if (state.autoScroll) {
        const view = $("#log-view");
        view.scrollTop = view.scrollHeight;
      }
    });
    $("#log-pause").addEventListener("change", (event) => { state.paused = event.target.checked; });
    $("#btn-log-clear").addEventListener("click", async () => {
      state.logs = [];
      rerenderLogs();
      try { await api("/api/logs/clear", { method: "POST" }); } catch { /* 忽略 */ }
    });
    $("#btn-log-download").addEventListener("click", () => {
      const text = state.logs.map((item) => `[${clock(item.ts)}] ${item.text}`).join("\r\n");
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `ninfer-serve-${new Date().toISOString().replace(/[:.]/g, "-")}.log`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 2000);
    });

    $("#btn-test-send").addEventListener("click", sendTest);

    document.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && $("#tab-test").classList.contains("active")) {
        sendTest();
      }
    });
  }

  init();
})();