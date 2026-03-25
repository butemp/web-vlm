/* ============================================================
   Video Intelligence Studio — Frontend Logic
   ============================================================ */

const state = {
  sourceId: null,
  runId: null,
  sourceKind: "",
  playbackUrl: "",
  sourceType: "upload",
  mode: "infer",
  theme: "dark",
  defaultPrompt: "",
  eventSource: null,
  hls: null,
  activeLogNode: null,
};

const el = {
  sourceTypeSeg: document.getElementById("sourceTypeSeg"),
  modeSeg: document.getElementById("modeSeg"),
  uploadPanel: document.getElementById("uploadPanel"),
  urlPanel: document.getElementById("urlPanel"),
  localPanel: document.getElementById("localPanel"),
  videoFile: document.getElementById("videoFile"),
  uploadBtn: document.getElementById("uploadBtn"),
  streamUrl: document.getElementById("streamUrl"),
  urlBtn: document.getElementById("urlBtn"),
  localPath: document.getElementById("localPath"),
  localBtn: document.getElementById("localBtn"),
  targetRow: document.getElementById("targetRow"),
  targetInput: document.getElementById("targetInput"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  streamPlayer: document.getElementById("streamPlayer"),
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
  themeToggleBtn: document.getElementById("themeToggleBtn"),
};

/* ── Helpers ── */
function setStatus(text, active = false) {
  el.statusText.textContent = text;
  el.statusChip.classList.toggle("active", active);
}

function setLive(active) {
  el.liveIndicator.classList.toggle("active", active);
}

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  const raw = await resp.text();

  let data = {};
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      const preview = raw.slice(0, 160).replace(/\s+/g, " ");
      throw new Error(`服务返回非 JSON（HTTP ${resp.status}）：${preview || "<empty>"}`);
    }
  }

  if (!resp.ok) {
    throw new Error(data.detail || data.message || `请求失败（HTTP ${resp.status}）`);
  }
  return data;
}

function isHlsUrl(url) {
  if (!url) return false;
  return url.toLowerCase().includes(".m3u8");
}

function stopNativePlayback() {
  if (state.hls) {
    try {
      state.hls.destroy();
    } catch {
      // ignore
    }
    state.hls = null;
  }
  if (el.streamPlayer) {
    try {
      el.streamPlayer.pause();
    } catch {
      // ignore
    }
    el.streamPlayer.removeAttribute("src");
    el.streamPlayer.load();
    el.streamPlayer.style.display = "none";
  }
}

async function tryStartDirectPlayback() {
  if (!el.streamPlayer) return false;

  const sourceKind = state.sourceKind || "";
  const playbackUrl = state.playbackUrl || "";
  if (!playbackUrl) return false;

  stopNativePlayback();
  const player = el.streamPlayer;

  // For uploaded files, try native playback first
  if (sourceKind === "file") {
    // Only use native playback in infer mode
    // In detect mode, we need MJPEG stream to show detection boxes
    if (state.mode !== "infer") {
      return false;
    }
    
    player.src = playbackUrl;
    player.style.display = "block";
    
    // Add error handler to fallback to MJPEG if native playback fails
    const errorHandler = () => {
      console.warn("Native video playback failed, will use MJPEG stream");
      player.style.display = "none";
      player.removeEventListener("error", errorHandler);
    };
    player.addEventListener("error", errorHandler);
    
    try {
      await player.play();
      // Remove error handler if play succeeds
      player.removeEventListener("error", errorHandler);
      return true;
    } catch (e) {
      console.warn("Native video play() failed:", e);
      // Don't return false immediately - the video might still load
      // Check if video has valid duration after a short delay
      await new Promise(r => setTimeout(r, 500));
      if (player.readyState >= 2) {
        return true;
      }
      player.style.display = "none";
      return false;
    }
  }

  // For HLS streams, ALWAYS use native playback (both infer and detect modes)
  // This avoids the slow OpenCV -> MJPEG conversion for H265 streams
  if (sourceKind === "url" && isHlsUrl(playbackUrl)) {
    if (player.canPlayType("application/vnd.apple.mpegurl")) {
      player.src = playbackUrl;
      player.style.display = "block";
      try {
        await player.play();
      } catch {
        // ignore
      }
      return true;
    }

    if (window.Hls && window.Hls.isSupported()) {
      const hls = new window.Hls({
        lowLatencyMode: true,
        backBufferLength: 10,
        maxBufferLength: 8,
        maxMaxBufferLength: 15,
        liveSyncDurationCount: 2,
        liveMaxLatencyDurationCount: 4,
        liveDurationInfinity: true,
        enableWorker: true,
        startLevel: -1,
      });
      hls.loadSource(playbackUrl);
      hls.attachMedia(player);
      hls.on(window.Hls.Events.MANIFEST_PARSED, () => {
        player.play().catch(() => {});
      });
      // Auto-sync to live edge on buffer stall
      hls.on(window.Hls.Events.ERROR, (event, data) => {
        if (data.type === window.Hls.ErrorTypes.MEDIA_ERROR) {
          hls.recoverMediaError();
        }
      });
      state.hls = hls;
      player.style.display = "block";
      return true;
    }
  }

  return false;
}

function applyTheme(theme) {
  state.theme = theme === "light" ? "light" : "dark";
  document.body.dataset.theme = state.theme;
  try {
    localStorage.setItem("web_vlm_theme", state.theme);
  } catch {
    // ignore
  }
  if (el.themeToggleBtn) {
    el.themeToggleBtn.textContent = state.theme === "dark" ? "切换浅色" : "切换深色";
  }
}

function toggleTheme() {
  applyTheme(state.theme === "dark" ? "light" : "dark");
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

function resetToInitialViewState() {
  state.sourceId = null;
  state.runId = null;
  state.sourceKind = "";
  state.playbackUrl = "";
  state.mode = "infer";
  state.sourceType = "upload";

  stopNativePlayback();
  el.streamView.removeAttribute("src");
  el.streamView.style.display = "none";
  el.streamPlaceholder.style.display = "flex";

  updateSourceTypeUI();
  updateModeUI();

  el.streamMeta.textContent = "等待输入视频源";
  el.promptInput.value = state.defaultPrompt;
  el.targetInput.value = "";
  el.streamUrl.value = "";
  el.localPath.value = "";
  if (el.videoFile) el.videoFile.value = "";
  if (el.fileDropText) el.fileDropText.textContent = "点击或拖拽视频文件";

  clearLogs();
  addLog("欢迎使用 Video Intelligence Studio。请先接入视频源，然后点击「开始分析」。");
  setStatus("等待连接", false);
  setLive(false);
  el.startBtn.disabled = true;
  el.stopBtn.disabled = true;
}

/* ── UI State ── */
function updateSourceTypeUI() {
  el.sourceTypeSeg.querySelectorAll("button").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.sourceType === state.sourceType);
  });
  el.uploadPanel.classList.toggle("hidden", state.sourceType !== "upload");
  el.urlPanel.classList.toggle("hidden", state.sourceType !== "url");
  el.localPanel.classList.toggle("hidden", state.sourceType !== "local");
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
async function stopAnalysis(notifyBackend = true, resetToInitial = false) {
  const sourceId = state.sourceId;
  const runId = state.runId;

  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }

  if (resetToInitial) {
    resetToInitialViewState();
  } else {
    state.runId = null;
    stopNativePlayback();
    el.streamView.removeAttribute("src");
    el.streamView.style.display = "none";
    el.streamPlaceholder.style.display = "flex";
    el.stopBtn.disabled = true;
    el.startBtn.disabled = !state.sourceId;
    el.streamMeta.textContent = "分析已暂停";
    setStatus("已停止", false);
    setLive(false);
  }

  if (notifyBackend && sourceId && runId) {
    fetchJson("/api/control/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId, run_id: runId }),
    }).catch((e) => {
      console.warn("stop control failed:", e);
    });
  }
}

async function startVideoStream() {
  if (!state.sourceId) {
    addLog("⚠ 请先上传视频或输入流 URL");
    return;
  }

  if (state.runId || state.eventSource) {
    await stopAnalysis(true);
  }

  const startData = await fetchJson("/api/control/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_id: state.sourceId }),
  });
  if (!startData.run_id) {
    throw new Error("后端未返回 run_id，请检查服务端日志");
  }
  state.runId = startData.run_id;

  clearLogs();

  const targets = encodeURIComponent(el.targetInput.value.trim());
  const streamUrl = `/api/stream/${state.sourceId}?run_id=${encodeURIComponent(state.runId)}&mode=${state.mode}&targets=${targets}&t=${Date.now()}`;
  const useDirectPlayback = await tryStartDirectPlayback();
  if (useDirectPlayback) {
    el.streamView.removeAttribute("src");
    el.streamView.style.display = "none";
    el.streamPlaceholder.style.display = "none";
  } else {
    stopNativePlayback();
    el.streamView.src = streamUrl;
    el.streamView.style.display = "block";
    el.streamPlaceholder.style.display = "none";
  }
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

async function uploadChunkWithRetry(url, formData, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const resp = await fetch(url, { method: "POST", body: formData });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }
      return await resp.json();
    } catch (err) {
      if (attempt === maxRetries) throw err;
      // Wait before retry: 1s, 2s, 3s...
      await new Promise(r => setTimeout(r, attempt * 1000));
    }
  }
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
  const originalText = el.uploadBtn.textContent;
  const CHUNK_SIZE = 2 * 1024 * 1024; // 2MB per chunk

  try {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    addLog(`⏳ 开始分片上传: ${file.name} (${sizeMB}MB, ${totalChunks} 片)`);

    // Step 1: Init upload session
    setStatus("初始化上传...", false);
    el.uploadBtn.textContent = "初始化...";
    const initData = await fetchJson("/api/upload/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        total_size: file.size,
        total_chunks: totalChunks,
      }),
    });
    const uploadId = initData.upload_id;

    // Step 2: Upload chunks with progress
    let uploadedChunks = 0;
    // Upload chunks with limited concurrency (2 at a time)
    const CONCURRENCY = 2;
    let nextChunk = 0;
    const errors = [];

    async function uploadNext() {
      while (nextChunk < totalChunks) {
        const idx = nextChunk++;
        const start = idx * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const blob = file.slice(start, end);

        const fd = new FormData();
        fd.append("file", blob, `chunk_${idx}`);

        try {
          await uploadChunkWithRetry(
            `/api/upload/chunk?upload_id=${encodeURIComponent(uploadId)}&chunk_index=${idx}`,
            fd
          );
          uploadedChunks++;
          const percent = Math.round((uploadedChunks / totalChunks) * 100);
          el.uploadBtn.textContent = `上传中 ${percent}%`;
          setStatus(`上传中 ${percent}% (${uploadedChunks}/${totalChunks})`, false);
        } catch (err) {
          errors.push({ idx, err });
        }
      }
    }

    const workers = [];
    for (let i = 0; i < Math.min(CONCURRENCY, totalChunks); i++) {
      workers.push(uploadNext());
    }
    await Promise.all(workers);

    if (errors.length > 0) {
      throw new Error(`${errors.length} 个分片上传失败，请重试`);
    }

    // Step 3: Complete / merge
    el.uploadBtn.textContent = "合并中...";
    setStatus("合并文件...", false);
    const data = await fetchJson("/api/upload/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: uploadId }),
    });

    state.sourceId = data.source_id;
    state.sourceKind = data.kind || "file";
    state.playbackUrl = data.playback_url || "";
    el.startBtn.disabled = false;
    el.streamMeta.textContent = `来源: ${data.name} (${data.size_mb || "?"}MB)`;
    setStatus("视频已就绪", true);
    addLog(`✓ 已加载: ${data.name} (${data.size_mb}MB)`);
  } finally {
    el.uploadBtn.disabled = false;
    el.uploadBtn.textContent = originalText;
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
    const data = await fetchJson("/api/source/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    state.sourceId = data.source_id;
    state.sourceKind = data.kind || "url";
    state.playbackUrl = data.playback_url || url;
    el.startBtn.disabled = false;
    el.streamMeta.textContent = `来源: ${url}`;
    setStatus("流地址已就绪", true);
    addLog(`✓ 已接入: ${url}`);
  } finally {
    el.urlBtn.disabled = false;
  }
}

async function loadLocalFile() {
  if (state.runId || state.eventSource) {
    await stopAnalysis(true);
  }

  const localPath = el.localPath.value.trim();
  if (!localPath) {
    addLog("⚠ 请输入服务器上的文件路径");
    return;
  }

  el.localBtn.disabled = true;
  setStatus("加载中...", false);

  try {
    const data = await fetchJson("/api/source/local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: localPath }),
    });

    state.sourceId = data.source_id;
    state.sourceKind = data.kind || "file";
    state.playbackUrl = data.playback_url || "";
    el.startBtn.disabled = false;
    el.streamMeta.textContent = `来源: ${data.name} (${data.size_mb || "?"}MB)`;
    setStatus("视频已就绪", true);
    addLog(`✓ 已加载服务器文件: ${data.name}`);
  } finally {
    el.localBtn.disabled = false;
  }
}

/* ── Boot ── */
async function boot() {
  try {
    const data = await fetchJson("/api/defaults");
    state.defaultPrompt = data.default_prompt || "请简单描述一下这个视频";
  } catch {
    state.defaultPrompt = "请简单描述一下这个视频";
  }

  el.promptInput.value = state.defaultPrompt;
  let preferredTheme = "dark";
  try {
    preferredTheme = localStorage.getItem("web_vlm_theme") || "dark";
  } catch {
    preferredTheme = "dark";
  }
  applyTheme(preferredTheme);
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

el.localBtn.addEventListener("click", async () => {
  try { await loadLocalFile(); }
  catch (err) { addLog(`❌ 加载失败: ${err.message}`, true); setStatus("加载失败", false); }
});

el.startBtn.addEventListener("click", async () => {
  try { await startVideoStream(); }
  catch (err) { addLog(`❌ 启动失败: ${err.message}`, true); setStatus("启动失败", false); }
});

el.stopBtn.addEventListener("click", async () => {
  await stopAnalysis(true, true);
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

if (el.themeToggleBtn) {
  el.themeToggleBtn.addEventListener("click", () => {
    toggleTheme();
  });
}

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
  stopNativePlayback();
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
