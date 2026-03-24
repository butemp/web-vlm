/* ============================================================
   Video Intelligence Studio — Frontend Logic
   ============================================================ */

const state = {
  sourceId: null,
  runId: null,
  sourceType: "upload",
  mode: "infer",
  defaultPrompt: "",
  eventSource: null,
  activeLogNode: null,
};

const el = {
  sourceTypeSeg: document.getElementById("sourceTypeSeg"),
  modeSeg: document.getElementById("modeSeg"),
  uploadPanel: document.getElementById("uploadPanel"),
  urlPanel: document.getElementById("urlPanel"),
  videoFile: document.getElementById("videoFile"),
  uploadBtn: document.getElementById("uploadBtn"),
  streamUrl: document.getElementById("streamUrl"),
  urlBtn: document.getElementById("urlBtn"),
  targetRow: document.getElementById("targetRow"),
  targetInput: document.getElementById("targetInput"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  streamView: document.getElementById("streamView"),
  streamPlaceholder: document.getElementById("streamPlaceholder"),
  statusChip: document.getElementById("statusChip"),
  statusText: document.getElementById("statusText"),
  streamMeta: document.getElementById("streamMeta"),
  promptArea: document.getElementById("promptArea"),
  promptInput: document.getElementById("promptInput"),
  applyPromptBtn: document.getElementById("applyPromptBtn"),
  resetPromptBtn: document.getElementById("resetPromptBtn"),
  logBox: document.getElementById("logBox"),
  insightTitle: document.getElementById("insightTitle"),
  modeTag: document.getElementById("modeTag"),
  liveIndicator: document.getElementById("liveIndicator"),
  fileDrop: document.getElementById("fileDrop"),
  fileDropText: document.getElementById("fileDropText"),
};

/* ── Helpers ── */
function setStatus(text, active = false) {
  el.statusText.textContent = text;
  el.statusChip.classList.toggle("active", active);
}

function setLive(active) {
  el.liveIndicator.classList.toggle("active", active);
}

function clearLogs() {
  el.logBox.innerHTML = "";
  state.activeLogNode = null;
}

function addLog(message, markLatest = false) {
  if (markLatest && state.activeLogNode) {
    state.activeLogNode.classList.remove("latest");
  }

  const item = document.createElement("div");
  item.className = "log-item" + (markLatest ? " latest" : "");

  const ts = document.createElement("div");
  ts.className = "log-ts";
  ts.textContent = new Date().toLocaleTimeString();

  const content = document.createElement("div");
  content.textContent = message;

  item.appendChild(ts);
  item.appendChild(content);
  el.logBox.appendChild(item);
  el.logBox.scrollTop = el.logBox.scrollHeight;

  if (markLatest) state.activeLogNode = item;
}

function appendToLatest(text) {
  if (!state.activeLogNode) {
    addLog(text, true);
    return;
  }
  const content = state.activeLogNode.lastElementChild;
  content.textContent += text;
  el.logBox.scrollTop = el.logBox.scrollHeight;
}

/* ── UI State ── */
function updateSourceTypeUI() {
  el.sourceTypeSeg.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.sourceType === state.sourceType);
  });
  el.uploadPanel.classList.toggle("hidden", state.sourceType !== "upload");
  el.urlPanel.classList.toggle("hidden", state.sourceType !== "url");
}

function updateModeUI() {
  el.modeSeg.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === state.mode);
  });
  const inferMode = state.mode === "infer";
  el.promptArea.classList.toggle("hidden", !inferMode);
  el.targetRow.classList.toggle("hidden", inferMode);
  el.insightTitle.textContent = inferMode ? "推理输出" : "检测事件";
  el.modeTag.textContent = inferMode ? "Qwen2.5-VL-3B" : "YOLOv8";
}

/* ── Core Actions ── */
async function stopAnalysis(notifyBackend = true) {
  const sourceId = state.sourceId;
  const runId = state.runId;

  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }

  if (notifyBackend && sourceId && runId) {
    try {
      await fetch("/api/control/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_id: sourceId, run_id: runId }),
      });
    } catch (e) {
      console.warn("stop control failed:", e);
    }
  }

  state.runId = null;
  el.streamView.removeAttribute("src");
  el.streamView.style.display = "none";
  el.streamPlaceholder.style.display = "flex";
  el.stopBtn.disabled = true;
  el.startBtn.disabled = !state.sourceId;
  el.streamMeta.textContent = "分析已暂停";
  setStatus("已停止", false);
  setLive(false);
}

async function startVideoStream() {
  if (!state.sourceId) {
    addLog("⚠ 请先上传视频或输入流 URL");
    return;
  }

  if (state.runId || state.eventSource) {
    await stopAnalysis(true);
  }

  const startResp = await fetch("/api/control/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_id: state.sourceId }),
  });
  const startData = await startResp.json();
  if (!startResp.ok) {
    throw new Error(startData.detail || "启动分析会话失败");
  }
  state.runId = startData.run_id;

  clearLogs();

  const targets = encodeURIComponent(el.targetInput.value.trim());
  const streamUrl = `/api/stream/${state.sourceId}?run_id=${encodeURIComponent(state.runId)}&mode=${state.mode}&targets=${targets}&t=${Date.now()}`;
  el.streamView.src = streamUrl;
  el.streamView.style.display = "block";
  el.streamPlaceholder.style.display = "none";
  el.streamMeta.textContent = state.mode === "infer"
    ? "Qwen2.5-VL-3B 正在流式推理"
    : "YOLOv8 正在实时检测";

  if (state.mode === "infer") {
    const prompt = encodeURIComponent(el.promptInput.value.trim() || state.defaultPrompt);
    state.eventSource = new EventSource(
      `/api/infer/stream?source_id=${state.sourceId}&run_id=${encodeURIComponent(state.runId)}&prompt=${prompt}`
    );
    state.eventSource.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        if (payload.type === "start") {
          addLog("", true);
        } else if (payload.type === "chunk") {
          appendToLatest(payload.text);
        } else if (payload.type === "error") {
          addLog("❌ " + payload.text, true);
        }
      } catch (e) {
        console.error("SSE parse error:", e);
      }
    };
    state.eventSource.onerror = () => {
      addLog("⚠ 推理流连接中断，请重新点击「开始分析」", true);
      setStatus("推理流中断", false);
      setLive(false);
      state.runId = null;
      el.startBtn.disabled = false;
      el.stopBtn.disabled = true;
    };
  } else {
    state.eventSource = new EventSource(
      `/api/detect/stream?source_id=${state.sourceId}&run_id=${encodeURIComponent(state.runId)}&targets=${encodeURIComponent(el.targetInput.value.trim())}`
    );
    state.eventSource.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        if (payload.type === "detect") {
          addLog(payload.text, true);
        }
      } catch (e) {
        console.error("SSE parse error:", e);
      }
    };
    state.eventSource.onerror = () => {
      addLog("⚠ 检测流连接中断，请重新点击「开始分析」", true);
      setStatus("检测流中断", false);
      setLive(false);
      state.runId = null;
      el.startBtn.disabled = false;
      el.stopBtn.disabled = true;
    };
  }

  el.startBtn.disabled = true;
  el.stopBtn.disabled = false;
  setStatus("分析中", true);
  setLive(true);
}

async function uploadVideo() {
  if (state.runId || state.eventSource) {
    await stopAnalysis(true);
  }

  const file = el.videoFile.files?.[0];
  if (!file) {
    addLog("⚠ 请先选择视频文件");
    return;
  }

  el.uploadBtn.disabled = true;
  el.uploadBtn.textContent = "上传中...";

  try {
    const fd = new FormData();
    fd.append("file", file);

    setStatus("上传中...", false);
    const resp = await fetch("/api/source/upload", { method: "POST", body: fd });
    const data = await resp.json();

    if (!resp.ok) throw new Error(data.detail || "上传失败");

    state.sourceId = data.source_id;
    el.startBtn.disabled = false;
    el.streamMeta.textContent = `来源: ${data.name} (${data.size_mb || "?"}MB)`;
    setStatus("视频已就绪", true);
    addLog(`✓ 已加载: ${data.name}`);
  } finally {
    el.uploadBtn.disabled = false;
    el.uploadBtn.textContent = "上传并加载";
  }
}

async function registerUrl() {
  if (state.runId || state.eventSource) {
    await stopAnalysis(true);
  }

  const url = el.streamUrl.value.trim();
  if (!url) {
    addLog("⚠ 请输入流媒体 URL");
    return;
  }

  el.urlBtn.disabled = true;
  setStatus("连接中...", false);

  try {
    const resp = await fetch("/api/source/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await resp.json();

    if (!resp.ok) throw new Error(data.detail || "URL 注册失败");

    state.sourceId = data.source_id;
    el.startBtn.disabled = false;
    el.streamMeta.textContent = `来源: ${url}`;
    setStatus("流地址已就绪", true);
    addLog(`✓ 已接入: ${url}`);
  } finally {
    el.urlBtn.disabled = false;
  }
}

/* ── Boot ── */
async function boot() {
  try {
    const resp = await fetch("/api/defaults");
    const data = await resp.json();
    state.defaultPrompt = data.default_prompt || "请简单描述一下这个视频";
  } catch {
    state.defaultPrompt = "请简单描述一下这个视频";
  }

  el.promptInput.value = state.defaultPrompt;
  updateSourceTypeUI();
  updateModeUI();
  el.startBtn.disabled = true;
  el.stopBtn.disabled = true;

  addLog("欢迎使用 Video Intelligence Studio。请先接入视频源，然后点击「开始分析」。");
}

/* ── Event Listeners ── */
el.sourceTypeSeg.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-source-type]");
  if (!btn) return;
  state.sourceType = btn.dataset.sourceType;
  updateSourceTypeUI();
});

el.modeSeg.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (!btn) return;

  if (state.runId || state.eventSource) {
    await stopAnalysis(true);
  }
  state.mode = btn.dataset.mode;
  updateModeUI();
  clearLogs();
  if (state.sourceId) el.startBtn.disabled = false;
});

el.uploadBtn.addEventListener("click", async () => {
  try { await uploadVideo(); }
  catch (err) { addLog(`❌ 上传失败: ${err.message}`, true); setStatus("上传失败", false); }
});

el.urlBtn.addEventListener("click", async () => {
  try { await registerUrl(); }
  catch (err) { addLog(`❌ 连接失败: ${err.message}`, true); setStatus("连接失败", false); }
});

el.startBtn.addEventListener("click", async () => {
  try { await startVideoStream(); }
  catch (err) { addLog(`❌ 启动失败: ${err.message}`, true); setStatus("启动失败", false); }
});

el.stopBtn.addEventListener("click", async () => {
  await stopAnalysis(true);
});

el.applyPromptBtn.addEventListener("click", async () => {
  if (state.mode !== "infer") return;
  if (!state.sourceId) { addLog("⚠ 请先接入视频源", true); return; }
  addLog("→ 已应用新 Prompt，重新开始流式推理...", true);
  await startVideoStream();
});

el.resetPromptBtn.addEventListener("click", () => {
  el.promptInput.value = state.defaultPrompt;
  addLog("→ Prompt 已恢复默认", true);
});

/* ── Drag & Drop ── */
el.fileDrop.addEventListener("dragover", (e) => {
  e.preventDefault();
  el.fileDrop.classList.add("dragover");
});
el.fileDrop.addEventListener("dragleave", () => {
  el.fileDrop.classList.remove("dragover");
});
el.fileDrop.addEventListener("drop", (e) => {
  e.preventDefault();
  el.fileDrop.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    el.videoFile.files = e.dataTransfer.files;
    el.fileDropText.textContent = e.dataTransfer.files[0].name;
  }
});
el.videoFile.addEventListener("change", () => {
  if (el.videoFile.files[0]) {
    el.fileDropText.textContent = el.videoFile.files[0].name;
  }
});

window.addEventListener("beforeunload", () => {
  if (state.eventSource) state.eventSource.close();
  if (state.sourceId && state.runId) {
    try {
      navigator.sendBeacon(
        "/api/control/stop",
        new Blob(
          [JSON.stringify({ source_id: state.sourceId, run_id: state.runId })],
          { type: "application/json" }
        )
      );
    } catch {
      // ignore
    }
  }
});

boot().catch((err) => {
  addLog(`❌ 初始化失败: ${err.message}`, true);
  setStatus("初始化失败", false);
});
