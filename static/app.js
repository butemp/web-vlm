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
  actionToken: 0,
  activeAction: "",
  uploadAbortController: null,
  mjpegStallTimer: null,
  mjpegLastNaturalWidth: 0,
  mjpegStallNotified: false,
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

async function fetchJsonWithTimeout(url, options = {}, timeoutMs = 15000) {
  const timeoutController = new AbortController();
  const externalSignal = options.signal || null;
  if (externalSignal) {
    if (externalSignal.aborted) {
      timeoutController.abort();
    } else {
      externalSignal.addEventListener("abort", () => timeoutController.abort(), { once: true });
    }
  }

  const timer = setTimeout(() => {
    timeoutController.abort();
  }, timeoutMs);

  try {
    return await fetchJson(url, { ...options, signal: timeoutController.signal });
  } catch (err) {
    if (timeoutController.signal.aborted && !(externalSignal && externalSignal.aborted)) {
      throw new Error(`请求超时（>${Math.floor(timeoutMs / 1000)}s）`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function isHlsUrl(url) {
  if (!url) return false;
  return url.toLowerCase().includes(".m3u8");
}

function stopMjpegStallDetection() {
  if (state.mjpegStallTimer) {
    clearInterval(state.mjpegStallTimer);
    state.mjpegStallTimer = null;
  }
  state.mjpegLastNaturalWidth = 0;
  state.mjpegStallNotified = false;
}

function startMjpegStallDetection() {
  stopMjpegStallDetection();
  let unchangedTicks = 0;
  // Poll the MJPEG img element: if its rendered size stays identical for
  // several seconds it likely means the backend stream stopped delivering.
  state.mjpegStallTimer = setInterval(() => {
    if (!state.runId || !el.streamView || el.streamView.style.display === "none") {
      return;
    }
    // Use naturalWidth change as a proxy — when the MJPEG stream is
    // alive the browser keeps decoding new JPEG frames.  We additionally
    // compare the raw element width which some browsers update per-frame.
    const w = el.streamView.naturalWidth || el.streamView.width || 0;
    if (w > 0 && w === state.mjpegLastNaturalWidth) {
      unchangedTicks++;
    } else {
      unchangedTicks = 0;
    }
    state.mjpegLastNaturalWidth = w;
    // ~8 seconds with no apparent frame change
    if (unchangedTicks >= 8 && !state.mjpegStallNotified) {
      state.mjpegStallNotified = true;
      addLog("⚠ 视频流可能已中断（画面长时间未更新），建议重新点击「开始分析」", true);
      setStatus("视频流可能已中断", false);
    }
  }, 1000);
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

function claimAction(actionName = "") {
  state.actionToken += 1;
  state.activeAction = actionName;
  return state.actionToken;
}

function isActionActive(token) {
  return token === state.actionToken;
}

function releaseAction(token) {
  if (isActionActive(token)) {
    state.activeAction = "";
  }
}

function isAbortLikeError(err) {
  if (!err) return false;
  const name = String(err.name || "");
  const msg = String(err.message || "").toLowerCase();
  return (
    name === "AbortError" ||
    msg.includes("aborted") ||
    msg.includes("abort")
  );
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
async function stopAnalysis(
  notifyBackend = true,
  resetToInitial = false,
  invalidateAction = true,
  waitBackendStop = false
) {
  if (invalidateAction) {
    claimAction("stop");
  }
  const sourceId = state.sourceId;
  const runId = state.runId;

  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  stopMjpegStallDetection();

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
    const stopPromise = fetchJsonWithTimeout("/api/control/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId, run_id: runId }),
    }, 7000).catch((e) => {
      console.warn("stop control failed:", e);
      return null;
    });
    if (waitBackendStop) {
      await stopPromise;
    }
  }
}

async function startVideoStream() {
  const actionToken = claimAction("start");
  const sourceId = state.sourceId;
  const mode = state.mode;

  if (!sourceId) {
    addLog("⚠ 请先上传视频或输入流 URL");
    releaseAction(actionToken);
    return;
  }

  if (state.runId || state.eventSource) {
    await stopAnalysis(true, false, false, true);
  }
  if (!isActionActive(actionToken)) return;

  const startData = await fetchJsonWithTimeout(
    "/api/control/start",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId }),
    },
    10000
  );
  if (!startData.run_id) {
    throw new Error("后端未返回 run_id，请检查服务端日志");
  }
  const runId = startData.run_id;

  if (!isActionActive(actionToken) || sourceId !== state.sourceId || mode !== state.mode) {
    bestEffortStopRun(sourceId, runId);
    return;
  }
  state.runId = runId;

  clearLogs();

  const targets = encodeURIComponent(el.targetInput.value.trim());
  const streamUrl = `/api/stream/${sourceId}?run_id=${encodeURIComponent(runId)}&mode=${mode}&targets=${targets}&t=${Date.now()}`;
  const useDirectPlayback = await tryStartDirectPlayback();
  if (!isActionActive(actionToken) || sourceId !== state.sourceId || runId !== state.runId) {
    bestEffortStopRun(sourceId, runId);
    stopNativePlayback();
    el.streamView.removeAttribute("src");
    return;
  }
  if (useDirectPlayback) {
    el.streamView.removeAttribute("src");
    el.streamView.style.display = "none";
    el.streamPlaceholder.style.display = "none";
  } else {
    stopNativePlayback();
    el.streamView.src = streamUrl;
    el.streamView.style.display = "block";
    el.streamPlaceholder.style.display = "none";
    startMjpegStallDetection();
  }
  el.streamMeta.textContent = mode === "infer"
    ? "Qwen2.5-VL-3B 正在流式推理"
    : "YOLOv8 正在实时检测";

  if (mode === "infer") {
    const prompt = encodeURIComponent(el.promptInput.value.trim() || state.defaultPrompt);
    function makeInferSSE() {
      return new EventSource(
        `/api/infer/stream?source_id=${sourceId}&run_id=${encodeURIComponent(runId)}&prompt=${prompt}`
      );
    }
    function onInferMessage(evt) {
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
    }
    state.eventSource = makeInferSSE();
    state.eventSource.onmessage = onInferMessage;
    const MAX_SSE_RECONNECTS = 3;
    let inferReconnects = 0;
    function attachInferErrorHandler() {
      state.eventSource.onerror = () => {
        if (inferReconnects < MAX_SSE_RECONNECTS && _is_run_active_local()) {
          inferReconnects++;
          const backoff = 1500 * inferReconnects;
          addLog(`⚠ 推理流短暂中断，正在重连(${inferReconnects}/${MAX_SSE_RECONNECTS})...`, true);
          if (state.eventSource) state.eventSource.close();
          setTimeout(() => {
            if (!_is_run_active_local()) return;
            state.eventSource = makeInferSSE();
            state.eventSource.onmessage = onInferMessage;
            attachInferErrorHandler();
          }, backoff);
          return;
        }
        addLog("⚠ 推理流连接中断，请重新点击「开始分析」", true);
        setStatus("推理流中断", false);
        setLive(false);
        state.runId = null;
        el.startBtn.disabled = !state.sourceId;
        el.stopBtn.disabled = true;
      };
    }
    attachInferErrorHandler();
  } else {
    function makeDetectSSE() {
      return new EventSource(
        `/api/detect/stream?source_id=${sourceId}&run_id=${encodeURIComponent(runId)}&targets=${encodeURIComponent(el.targetInput.value.trim())}`
      );
    }
    function onDetectMessage(evt) {
      try {
        const payload = JSON.parse(evt.data);
        if (payload.type === "detect") {
          addLog(payload.text, true);
        }
      } catch (e) {
        console.error("SSE parse error:", e);
      }
    }
    state.eventSource = makeDetectSSE();
    state.eventSource.onmessage = onDetectMessage;
    const MAX_DETECT_RECONNECTS = 3;
    let detectReconnects = 0;
    function attachDetectErrorHandler() {
      state.eventSource.onerror = () => {
        if (detectReconnects < MAX_DETECT_RECONNECTS && _is_run_active_local()) {
          detectReconnects++;
          const backoff = 1500 * detectReconnects;
          addLog(`⚠ 检测流短暂中断，正在重连(${detectReconnects}/${MAX_DETECT_RECONNECTS})...`, true);
          if (state.eventSource) state.eventSource.close();
          setTimeout(() => {
            if (!_is_run_active_local()) return;
            state.eventSource = makeDetectSSE();
            state.eventSource.onmessage = onDetectMessage;
            attachDetectErrorHandler();
          }, backoff);
          return;
        }
        addLog("⚠ 检测流连接中断，请重新点击「开始分析」", true);
        setStatus("检测流中断", false);
        setLive(false);
        state.runId = null;
        el.startBtn.disabled = !state.sourceId;
        el.stopBtn.disabled = true;
      };
    }
    attachDetectErrorHandler();
  }

  if (!isActionActive(actionToken)) {
    bestEffortStopRun(sourceId, runId);
    return;
  }
  el.startBtn.disabled = true;
  el.stopBtn.disabled = false;
  setStatus("分析中", true);
  setLive(true);
  releaseAction(actionToken);
}

function _is_run_active_local() {
  return !!(state.sourceId && state.runId);
}

function bestEffortStopRun(sourceId, runId) {
  if (!sourceId || !runId) return;
  fetchJson("/api/control/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_id: sourceId, run_id: runId }),
  }).catch(() => {});
}

async function uploadChunkWithRetry(url, formData, maxRetries = 3, signal = null) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    let timer = null;
    try {
      const timeoutController = new AbortController();
      if (signal) {
        if (signal.aborted) timeoutController.abort();
        else signal.addEventListener("abort", () => timeoutController.abort(), { once: true });
      }
      timer = setTimeout(() => timeoutController.abort(), 20000);
      const resp = await fetch(url, {
        method: "POST",
        body: formData,
        signal: timeoutController.signal,
      });
      clearTimeout(timer);
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }
      return await resp.json();
    } catch (err) {
      if (timer) clearTimeout(timer);
      if (signal && signal.aborted) {
        throw new Error("上传中断：检测到长时间卡住，请重新上传");
      }
      if (isAbortLikeError(err) && attempt === maxRetries) {
        throw new Error("上传超时：网络可能不稳定，请重新上传");
      }
      if (attempt === maxRetries) throw err;
      // Wait before retry: 1s, 2s, 3s...
      await new Promise(r => setTimeout(r, attempt * 1000));
    }
  }
}

async function uploadVideo() {
  const actionToken = claimAction("upload");
  if (state.runId || state.eventSource) {
    await stopAnalysis(true, false, false, true);
  }
  if (!isActionActive(actionToken)) return;

  const file = el.videoFile.files?.[0];
  if (!file) {
    addLog("⚠ 请先选择视频文件");
    releaseAction(actionToken);
    return;
  }

  el.uploadBtn.disabled = true;
  const originalText = el.uploadBtn.textContent;
  const CHUNK_SIZE = 2 * 1024 * 1024; // 2MB per chunk
  let watchdog = null;
  let abortController = null;

  try {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    addLog(`⏳ 开始分片上传: ${file.name} (${sizeMB}MB, ${totalChunks} 片)`);
    const uploadStartTs = Date.now();
    let lastProgressTs = Date.now();
    const fileMB = file.size / (1024 * 1024);
    const noProgressTimeoutMs = 60000;
    const hardTimeoutMs = Math.min(20 * 60 * 1000, Math.max(2 * 60 * 1000, Math.floor(fileMB * 4500)));
    abortController = new AbortController();
    state.uploadAbortController = abortController;
    watchdog = setInterval(() => {
      const now = Date.now();
      if ((now - lastProgressTs) > noProgressTimeoutMs || (now - uploadStartTs) > hardTimeoutMs) {
        abortController.abort();
      }
    }, 2000);

    // Step 1: Init upload session
    setStatus("初始化上传...", false);
    el.uploadBtn.textContent = "初始化...";
    let initData = null;
    const initPayload = JSON.stringify({
      filename: file.name,
      total_size: file.size,
      total_chunks: totalChunks,
    });
    const INIT_RETRIES = 3;
    for (let attempt = 1; attempt <= INIT_RETRIES; attempt++) {
      try {
        initData = await fetchJsonWithTimeout(
          "/api/upload/init",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: initPayload,
            signal: abortController.signal,
          },
          12000
        );
        if (!isActionActive(actionToken)) return;
        lastProgressTs = Date.now();
        break;
      } catch (err) {
        if (abortController.signal.aborted) {
          throw new Error("上传初始化卡住时间过长，请重新上传");
        }
        if (attempt === INIT_RETRIES) throw err;
        const waitMs = 700 * attempt;
        setStatus(`初始化重试中（${attempt}/${INIT_RETRIES - 1}）...`, false);
        el.uploadBtn.textContent = `重试 ${attempt}/${INIT_RETRIES - 1}`;
        await new Promise((r) => setTimeout(r, waitMs));
        if (!isActionActive(actionToken)) return;
      }
    }
    if (!initData || !initData.upload_id) {
      throw new Error("初始化上传失败：后端未返回 upload_id");
    }
    const uploadId = initData.upload_id;

    // Step 2: Upload chunks with progress
    let uploadedChunks = 0;
    // Upload chunks with limited concurrency (2 at a time)
    const CONCURRENCY = 2;
    let nextChunk = 0;
    const errors = [];

    async function uploadNext() {
      while (nextChunk < totalChunks) {
        if (!isActionActive(actionToken)) return;
        const idx = nextChunk++;
        const start = idx * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const blob = file.slice(start, end);

        const fd = new FormData();
        fd.append("file", blob, `chunk_${idx}`);

        try {
          await uploadChunkWithRetry(
            `/api/upload/chunk?upload_id=${encodeURIComponent(uploadId)}&chunk_index=${idx}`,
            fd,
            3,
            abortController.signal,
          );
          uploadedChunks++;
          lastProgressTs = Date.now();
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
    if (!isActionActive(actionToken)) return;

    if (errors.length > 0) {
      throw new Error(`${errors.length} 个分片上传失败，请重试`);
    }

    // Step 3: Complete / merge
    el.uploadBtn.textContent = "合并中...";
    setStatus("合并文件...", false);
    let data = null;
    const COMPLETE_RETRIES = 2;
    for (let attempt = 1; attempt <= COMPLETE_RETRIES; attempt++) {
      try {
        data = await fetchJsonWithTimeout(
          "/api/upload/complete",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ upload_id: uploadId }),
            signal: abortController.signal,
          },
          30000
        );
        lastProgressTs = Date.now();
        break;
      } catch (err) {
        if (abortController.signal.aborted) {
          throw new Error("上传合并阶段超时，请重新上传");
        }
        if (attempt === COMPLETE_RETRIES) throw err;
        await new Promise((r) => setTimeout(r, 1200));
      }
    }
    if (!data) {
      throw new Error("上传合并失败，请重新上传");
    }
    if (!isActionActive(actionToken)) return;

    state.sourceId = data.source_id;
    state.sourceKind = data.kind || "file";
    state.playbackUrl = data.playback_url || "";
    el.startBtn.disabled = false;
    el.streamMeta.textContent = `来源: ${data.name} (${data.size_mb || "?"}MB)`;
    setStatus("视频已就绪", true);
    addLog(`✓ 已加载: ${data.name} (${data.size_mb}MB)`);
    // 视觉反馈：短暂高亮开始按钮
    el.startBtn.classList.add("pulse-ready");
    setTimeout(() => el.startBtn.classList.remove("pulse-ready"), 2000);
  } finally {
    if (watchdog) {
      clearInterval(watchdog);
      watchdog = null;
    }
    if (state.uploadAbortController) {
      try { state.uploadAbortController.abort(); } catch {}
      state.uploadAbortController = null;
    }
    el.uploadBtn.disabled = false;
    el.uploadBtn.textContent = originalText;
    releaseAction(actionToken);
  }
}

async function registerUrl() {
  const actionToken = claimAction("connect_url");
  if (state.runId || state.eventSource) {
    await stopAnalysis(true, false, false, true);
  }
  if (!isActionActive(actionToken)) return;

  const url = el.streamUrl.value.trim();
  if (!url) {
    addLog("⚠ 请输入流媒体 URL");
    releaseAction(actionToken);
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
    if (!isActionActive(actionToken)) return;

    state.sourceId = data.source_id;
    state.sourceKind = data.kind || "url";
    state.playbackUrl = data.playback_url || url;
    el.startBtn.disabled = false;
    el.streamMeta.textContent = `来源: ${url}`;
    setStatus("流地址已就绪", true);
    addLog(`✓ 已接入: ${url}`);
    el.startBtn.classList.add("pulse-ready");
    setTimeout(() => el.startBtn.classList.remove("pulse-ready"), 2000);
  } finally {
    el.urlBtn.disabled = false;
    releaseAction(actionToken);
  }
}

async function loadLocalFile() {
  const actionToken = claimAction("load_local");
  if (state.runId || state.eventSource) {
    await stopAnalysis(true, false, false, true);
  }
  if (!isActionActive(actionToken)) return;

  const localPath = el.localPath.value.trim();
  if (!localPath) {
    addLog("⚠ 请输入服务器上的文件路径");
    releaseAction(actionToken);
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
    if (!isActionActive(actionToken)) return;

    state.sourceId = data.source_id;
    state.sourceKind = data.kind || "file";
    state.playbackUrl = data.playback_url || "";
    el.startBtn.disabled = false;
    el.streamMeta.textContent = `来源: ${data.name} (${data.size_mb || "?"}MB)`;
    setStatus("视频已就绪", true);
    addLog(`✓ 已加载服务器文件: ${data.name}`);
    el.startBtn.classList.add("pulse-ready");
    setTimeout(() => el.startBtn.classList.remove("pulse-ready"), 2000);
  } finally {
    el.localBtn.disabled = false;
    releaseAction(actionToken);
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
el.sourceTypeSeg.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-source-type]");
  if (!btn) return;
  if (state.runId || state.eventSource) {
    await stopAnalysis(true, false, true, true);
  }
  state.sourceType = btn.dataset.sourceType;
  updateSourceTypeUI();
});

el.modeSeg.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (!btn) return;

  if (state.runId || state.eventSource) {
    await stopAnalysis(true, false, true, true);
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
  const origText = el.startBtn.textContent;
  el.startBtn.textContent = "启动中...";
  el.startBtn.disabled = true;
  try {
    await startVideoStream();
  } catch (err) {
    addLog(`❌ 启动失败: ${err.message}`, true);
    setStatus("启动失败", false);
  } finally {
    el.startBtn.textContent = origText;
    el.startBtn.disabled = !!state.runId || !state.sourceId;
  }
});

el.stopBtn.addEventListener("click", async () => {
  await stopAnalysis(true, false);
  addLog("■ 分析已停止，可重新点击「开始分析」继续");
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
