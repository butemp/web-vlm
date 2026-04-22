/* ============================================================
   Video Intelligence Studio — Frontend Logic (Dual Panel)
   ============================================================ */

const PANEL_COUNT = 2;

const state = {
  sourceType: "upload",
  mode: "infer",
  theme: "dark",
  defaultPrompt: "",
  actionToken: 0,
  activeAction: "",
  uploadAbortController: null,
  panels: [],
};

// Initialize per-panel state
for (let i = 0; i < PANEL_COUNT; i++) {
  state.panels.push({
    sourceId: null,
    runId: null,
    sourceKind: "",
    playbackUrl: "",
    eventSource: null,
    hls: null,
    activeLogNode: null,
    mjpegStallTimer: null,
    mjpegLastNaturalWidth: 0,
    mjpegStallNotified: false,
  });
}

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
  statusChip: document.getElementById("statusChip"),
  statusText: document.getElementById("statusText"),
  promptArea: document.getElementById("promptArea"),
  promptInput: document.getElementById("promptInput"),
  applyPromptBtn: document.getElementById("applyPromptBtn"),
  resetPromptBtn: document.getElementById("resetPromptBtn"),
  insightTitle: document.getElementById("insightTitle"),
  modeTag: document.getElementById("modeTag"),
  fileDrop: document.getElementById("fileDrop"),
  fileDropText: document.getElementById("fileDropText"),
  uploadFeedback: document.getElementById("uploadFeedback"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
  panels: [],
};

// Initialize per-panel DOM elements
for (let i = 0; i < PANEL_COUNT; i++) {
  el.panels.push({
    panel: document.getElementById(`videoPanel${i}`),
    streamPlayer: document.getElementById(`streamPlayer${i}`),
    streamView: document.getElementById(`streamView${i}`),
    streamPlaceholder: document.getElementById(`streamPlaceholder${i}`),
    streamMeta: document.getElementById(`streamMeta${i}`),
    liveIndicator: document.getElementById(`liveIndicator${i}`),
    logBox: document.getElementById(`logBox${i}`),
    sourceName: document.getElementById(`sourceName${i}`),
  });
}

/* ── Helpers ── */
function setStatus(text, active = false) {
  el.statusText.textContent = text;
  el.statusChip.classList.toggle("active", active);
}

function setLive(active, panelIdx) {
  if (panelIdx === undefined) {
    for (let i = 0; i < PANEL_COUNT; i++) {
      el.panels[i].liveIndicator.classList.toggle("active", active);
    }
  } else {
    el.panels[panelIdx].liveIndicator.classList.toggle("active", active);
  }
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

function stopMjpegStallDetection(panelIdx) {
  const p = state.panels[panelIdx];
  if (p.mjpegStallTimer) {
    clearInterval(p.mjpegStallTimer);
    p.mjpegStallTimer = null;
  }
  p.mjpegLastNaturalWidth = 0;
  p.mjpegStallNotified = false;
}

function startMjpegStallDetection(panelIdx) {
  stopMjpegStallDetection(panelIdx);
  const p = state.panels[panelIdx];
  const pe = el.panels[panelIdx];
  let unchangedTicks = 0;
  p.mjpegStallTimer = setInterval(() => {
    if (!p.runId || !pe.streamView || pe.streamView.style.display === "none") {
      return;
    }
    const w = pe.streamView.naturalWidth || pe.streamView.width || 0;
    if (w > 0 && w === p.mjpegLastNaturalWidth) {
      unchangedTicks++;
    } else {
      unchangedTicks = 0;
    }
    p.mjpegLastNaturalWidth = w;
    if (unchangedTicks >= 8 && !p.mjpegStallNotified) {
      p.mjpegStallNotified = true;
      addLog("⚠ 视频流可能已中断（画面长时间未更新），建议重新点击「开始分析」", panelIdx, true);
      setStatus("视频流可能已中断", false);
    }
  }, 1000);
}

function stopNativePlayback(panelIdx) {
  const p = state.panels[panelIdx];
  const pe = el.panels[panelIdx];
  if (p.hls) {
    try {
      p.hls.destroy();
    } catch {
      // ignore
    }
    p.hls = null;
  }
  if (pe.streamPlayer) {
    try {
      pe.streamPlayer.pause();
    } catch {
      // ignore
    }
    pe.streamPlayer.removeAttribute("src");
    pe.streamPlayer.load();
    pe.streamPlayer.style.display = "none";
  }
}

async function attemptPlayerStart(player, timeoutMs = 2500) {
  return await new Promise((resolve) => {
    let settled = false;
    let timer = null;

    function cleanup() {
      if (timer) clearTimeout(timer);
      player.removeEventListener("loadeddata", onLoadedData);
      player.removeEventListener("canplay", onCanPlay);
      player.removeEventListener("playing", onPlaying);
      player.removeEventListener("error", onError);
    }

    function finish(ok) {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(ok);
    }

    function onLoadedData() {
      finish(true);
    }

    function onCanPlay() {
      finish(true);
    }

    function onPlaying() {
      finish(true);
    }

    function onError() {
      finish(false);
    }

    player.addEventListener("loadeddata", onLoadedData);
    player.addEventListener("canplay", onCanPlay);
    player.addEventListener("playing", onPlaying);
    player.addEventListener("error", onError);

    timer = setTimeout(() => finish(false), timeoutMs);

    try {
      const playPromise = player.play();
      if (playPromise && typeof playPromise.then === "function") {
        playPromise.then(() => finish(true)).catch(() => finish(false));
      }
    } catch {
      finish(false);
    }
  });
}

async function tryStartDirectPlayback(panelIdx) {
  const p = state.panels[panelIdx];
  const pe = el.panels[panelIdx];
  if (!pe.streamPlayer) return false;

  // Reliability first: infer mode uses backend MJPEG so display and inference
  // stay on the same server-side decode path.
  if (state.mode === "infer") {
    return false;
  }

  const sourceKind = p.sourceKind || "";
  const playbackUrl = p.playbackUrl || "";
  if (!playbackUrl) return false;

  stopNativePlayback(panelIdx);
  const player = pe.streamPlayer;

  if (sourceKind === "file") {
    if (state.mode !== "infer") {
      return false;
    }

    player.src = playbackUrl;
    player.style.display = "block";
    const started = await attemptPlayerStart(player, 2500);
    if (started || player.readyState >= 2) {
      return true;
    }
    console.warn(`Panel ${panelIdx}: Native file playback timed out, fallback to MJPEG`);
    player.style.display = "none";
    return false;
  }

  if (sourceKind === "url" && isHlsUrl(playbackUrl)) {
    if (player.canPlayType("application/vnd.apple.mpegurl")) {
      player.src = playbackUrl;
      player.style.display = "block";
      const started = await attemptPlayerStart(player, 2500);
      if (started || player.readyState >= 2) {
        return true;
      }
      player.style.display = "none";
      return false;
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
      hls.on(window.Hls.Events.ERROR, (event, data) => {
        if (data.type === window.Hls.ErrorTypes.MEDIA_ERROR) {
          hls.recoverMediaError();
        }
      });
      p.hls = hls;
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

function clearLogs(panelIdx) {
  el.panels[panelIdx].logBox.innerHTML = "";
  state.panels[panelIdx].activeLogNode = null;
}

function addLog(message, panelIdx, markLatest = false) {
  const p = state.panels[panelIdx];
  const logBox = el.panels[panelIdx].logBox;

  if (markLatest && p.activeLogNode) {
    p.activeLogNode.classList.remove("latest");
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
  logBox.appendChild(item);
  logBox.scrollTop = logBox.scrollHeight;

  if (markLatest) p.activeLogNode = item;
}

function appendToLatest(text, panelIdx) {
  const p = state.panels[panelIdx];
  const logBox = el.panels[panelIdx].logBox;
  if (!p.activeLogNode) {
    addLog(text, panelIdx, true);
    return;
  }
  const content = p.activeLogNode.lastElementChild;
  content.textContent += text;
  logBox.scrollTop = logBox.scrollHeight;
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

function _is_run_active_local(panelIdx) {
  if (panelIdx === undefined) {
    return state.panels.some(p => !!(p.sourceId && p.runId));
  }
  const p = state.panels[panelIdx];
  return !!(p.sourceId && p.runId);
}

function getNextAvailablePanel() {
  if (!state.panels[0].sourceId) return 0;
  if (!state.panels[1].sourceId) return 1;
  return 0; // replace oldest
}

function getTargetPanelsForFiles(fileCount) {
  const emptyPanels = [];
  const occupiedPanels = [];

  for (let i = 0; i < PANEL_COUNT; i++) {
    if (state.panels[i].sourceId) {
      occupiedPanels.push(i);
    } else {
      emptyPanels.push(i);
    }
  }

  return [...emptyPanels, ...occupiedPanels].slice(0, fileCount);
}

function describeSelectedFiles(files) {
  if (!files || files.length === 0) {
    return "点击或拖拽一个或多个视频文件";
  }
  if (files.length === 1) {
    return files[0].name;
  }
  return `已选择 ${files.length} 个文件`;
}

function updateSelectedFilesLabel(files) {
  if (!el.fileDropText) return;
  el.fileDropText.textContent = describeSelectedFiles(files);
}

function setUploadFeedback(text = "", visible = true) {
  if (!el.uploadFeedback) return;
  el.uploadFeedback.textContent = text;
  el.uploadFeedback.classList.toggle("hidden", !visible || !text);
}

function updateButtonStates() {
  const anySource = state.panels.some(p => p.sourceId);
  const anyRunning = state.panels.some(p => p.runId);
  el.startBtn.disabled = anyRunning || !anySource;
  el.stopBtn.disabled = !anyRunning;
}

function bestEffortStopRun(sourceId, runId) {
  if (!sourceId || !runId) return;
  fetchJson("/api/control/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_id: sourceId, run_id: runId }),
  }).catch(() => {});
}

function resetPanelViewState(panelIdx) {
  const p = state.panels[panelIdx];
  const pe = el.panels[panelIdx];

  p.sourceId = null;
  p.runId = null;
  p.sourceKind = "";
  p.playbackUrl = "";

  stopNativePlayback(panelIdx);
  pe.streamView.removeAttribute("src");
  pe.streamView.style.display = "none";
  pe.streamPlaceholder.style.display = "flex";
  pe.streamMeta.textContent = "等待输入视频源";
  pe.sourceName.textContent = "";
  clearLogs(panelIdx);
  setLive(false, panelIdx);
}

function resetToInitialViewState() {
  state.mode = "infer";
  state.sourceType = "upload";

  for (let i = 0; i < PANEL_COUNT; i++) {
    resetPanelViewState(i);
    addLog("欢迎使用 Video Intelligence Studio。请先接入视频源，然后点击「开始分析」。", i);
  }

  updateSourceTypeUI();
  updateModeUI();

  el.promptInput.value = state.defaultPrompt;
  el.targetInput.value = "";
  el.streamUrl.value = "";
  el.localPath.value = "";
  if (el.videoFile) el.videoFile.value = "";
  updateSelectedFilesLabel([]);
  setUploadFeedback("", false);

  setStatus("等待连接", false);
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

/* ── Core Actions: Per-Panel ── */
async function stopAnalysisForPanel(panelIdx, notifyBackend = true, waitBackendStop = false) {
  const p = state.panels[panelIdx];
  const pe = el.panels[panelIdx];
  const sourceId = p.sourceId;
  const runId = p.runId;

  if (p.eventSource) {
    p.eventSource.close();
    p.eventSource = null;
  }
  stopMjpegStallDetection(panelIdx);

  p.runId = null;
  stopNativePlayback(panelIdx);
  pe.streamView.removeAttribute("src");
  pe.streamView.style.display = "none";
  pe.streamPlaceholder.style.display = "flex";
  pe.streamMeta.textContent = p.sourceId ? "分析已暂停" : "等待输入视频源";
  setLive(false, panelIdx);

  if (notifyBackend && sourceId && runId) {
    const stopPromise = fetchJsonWithTimeout("/api/control/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId, run_id: runId }),
    }, 7000).catch((e) => {
      console.warn(`Panel ${panelIdx}: stop control failed:`, e);
      return null;
    });
    if (waitBackendStop) {
      await stopPromise;
    }
  }
}

async function stopAnalysis(
  notifyBackend = true,
  resetToInitial = false,
  invalidateAction = true,
  waitBackendStop = false
) {
  if (invalidateAction) {
    claimAction("stop");
  }

  if (resetToInitial) {
    // Stop all panels first, then reset
    for (let i = 0; i < PANEL_COUNT; i++) {
      await stopAnalysisForPanel(i, notifyBackend, waitBackendStop);
    }
    resetToInitialViewState();
  } else {
    for (let i = 0; i < PANEL_COUNT; i++) {
      await stopAnalysisForPanel(i, notifyBackend, waitBackendStop);
    }
    setStatus("已停止", false);
    updateButtonStates();
  }
}

async function startPanelStream(panelIdx, actionToken, mode) {
  const p = state.panels[panelIdx];
  const pe = el.panels[panelIdx];

  if (!p.sourceId) return;

  // Stop existing run for this panel if any
  if (p.runId || p.eventSource) {
    await stopAnalysisForPanel(panelIdx, true, true);
  }
  if (!isActionActive(actionToken)) return;

  const sourceId = p.sourceId;
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
    throw new Error(`面板 ${panelIdx + 1}: 后端未返回 run_id，请检查服务端日志`);
  }
  const runId = startData.run_id;

  if (!isActionActive(actionToken) || sourceId !== p.sourceId || mode !== state.mode) {
    bestEffortStopRun(sourceId, runId);
    return;
  }
  p.runId = runId;

  clearLogs(panelIdx);

  const targets = encodeURIComponent(el.targetInput.value.trim());
  const streamUrl = `/api/stream/${sourceId}?run_id=${encodeURIComponent(runId)}&mode=${mode}&targets=${targets}&t=${Date.now()}`;
  const useDirectPlayback = await tryStartDirectPlayback(panelIdx);
  if (!isActionActive(actionToken) || sourceId !== p.sourceId || runId !== p.runId) {
    bestEffortStopRun(sourceId, runId);
    stopNativePlayback(panelIdx);
    pe.streamView.removeAttribute("src");
    return;
  }
  if (useDirectPlayback) {
    pe.streamView.removeAttribute("src");
    pe.streamView.style.display = "none";
    pe.streamPlaceholder.style.display = "none";
  } else {
    stopNativePlayback(panelIdx);
    pe.streamView.src = streamUrl;
    pe.streamView.style.display = "block";
    pe.streamPlaceholder.style.display = "none";
    startMjpegStallDetection(panelIdx);
  }
  pe.streamMeta.textContent = mode === "infer"
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
          addLog("", panelIdx, true);
        } else if (payload.type === "chunk") {
          appendToLatest(payload.text, panelIdx);
        } else if (payload.type === "error") {
          addLog("❌ " + payload.text, panelIdx, true);
        }
      } catch (e) {
        console.error(`Panel ${panelIdx}: SSE parse error:`, e);
      }
    }
    p.eventSource = makeInferSSE();
    p.eventSource.onmessage = onInferMessage;
    const MAX_SSE_RECONNECTS = 3;
    let inferReconnects = 0;
    function attachInferErrorHandler() {
      p.eventSource.onerror = () => {
        if (inferReconnects < MAX_SSE_RECONNECTS && _is_run_active_local(panelIdx)) {
          inferReconnects++;
          const backoff = 1500 * inferReconnects;
          addLog(`⚠ 推理流短暂中断，正在重连(${inferReconnects}/${MAX_SSE_RECONNECTS})...`, panelIdx, true);
          if (p.eventSource) p.eventSource.close();
          setTimeout(() => {
            if (!_is_run_active_local(panelIdx)) return;
            p.eventSource = makeInferSSE();
            p.eventSource.onmessage = onInferMessage;
            attachInferErrorHandler();
          }, backoff);
          return;
        }
        addLog("⚠ 推理流连接中断，请重新点击「开始分析」", panelIdx, true);
        setStatus("推理流中断", false);
        setLive(false, panelIdx);
        p.runId = null;
        updateButtonStates();
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
          addLog(payload.text, panelIdx, true);
        }
      } catch (e) {
        console.error(`Panel ${panelIdx}: SSE parse error:`, e);
      }
    }
    p.eventSource = makeDetectSSE();
    p.eventSource.onmessage = onDetectMessage;
    const MAX_DETECT_RECONNECTS = 3;
    let detectReconnects = 0;
    function attachDetectErrorHandler() {
      p.eventSource.onerror = () => {
        if (detectReconnects < MAX_DETECT_RECONNECTS && _is_run_active_local(panelIdx)) {
          detectReconnects++;
          const backoff = 1500 * detectReconnects;
          addLog(`⚠ 检测流短暂中断，正在重连(${detectReconnects}/${MAX_DETECT_RECONNECTS})...`, panelIdx, true);
          if (p.eventSource) p.eventSource.close();
          setTimeout(() => {
            if (!_is_run_active_local(panelIdx)) return;
            p.eventSource = makeDetectSSE();
            p.eventSource.onmessage = onDetectMessage;
            attachDetectErrorHandler();
          }, backoff);
          return;
        }
        addLog("⚠ 检测流连接中断，请重新点击「开始分析」", panelIdx, true);
        setStatus("检测流中断", false);
        setLive(false, panelIdx);
        p.runId = null;
        updateButtonStates();
      };
    }
    attachDetectErrorHandler();
  }

  setLive(true, panelIdx);
}

async function startAllStreams() {
  const actionToken = claimAction("start");
  const mode = state.mode;
  const activePanels = [];

  for (let i = 0; i < PANEL_COUNT; i++) {
    if (state.panels[i].sourceId) {
      activePanels.push(i);
    }
  }

  if (activePanels.length === 0) {
    addLog("⚠ 请先上传视频或输入流 URL", 0);
    releaseAction(actionToken);
    return;
  }

  // Stop any currently running panels first
  for (let i = 0; i < PANEL_COUNT; i++) {
    if (state.panels[i].runId || state.panels[i].eventSource) {
      await stopAnalysisForPanel(i, true, true);
    }
  }
  if (!isActionActive(actionToken)) return;

  for (const panelIdx of activePanels) {
    if (!isActionActive(actionToken)) return;
    try {
      await startPanelStream(panelIdx, actionToken, mode);
    } catch (err) {
      addLog(`❌ 面板 ${panelIdx + 1} 启动失败: ${err.message}`, panelIdx, true);
    }
  }

  if (!isActionActive(actionToken)) return;
  updateButtonStates();
  if (state.panels.some(p => p.runId)) {
    setStatus("分析中", true);
  }
  releaseAction(actionToken);
}

/* ── Upload / Register / Load ── */
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
      await new Promise(r => setTimeout(r, attempt * 1000));
    }
  }
}

async function uploadSingleVideoToPanel(file, panelIdx, actionToken, fileIndex, totalFiles) {
  const p = state.panels[panelIdx];
  const pe = el.panels[panelIdx];
  const CHUNK_SIZE = 2 * 1024 * 1024;
  let watchdog = null;
  let abortController = null;

  try {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    const batchPrefix = totalFiles > 1 ? `[${fileIndex + 1}/${totalFiles}] ` : "";
    addLog(
      `⏳ ${batchPrefix}开始分片上传: ${file.name} (${sizeMB}MB, ${totalChunks} 片)`,
      panelIdx
    );
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
    setStatus(`初始化上传 ${fileIndex + 1}/${totalFiles}...`, false);
    el.uploadBtn.textContent = totalFiles > 1 ? `初始化 ${fileIndex + 1}/${totalFiles}` : "初始化...";
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
        setStatus(`初始化重试 ${fileIndex + 1}/${totalFiles}（${attempt}/${INIT_RETRIES - 1}）...`, false);
        el.uploadBtn.textContent = `重试 ${fileIndex + 1}/${totalFiles}`;
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
          el.uploadBtn.textContent = totalFiles > 1
            ? `上传 ${fileIndex + 1}/${totalFiles} ${percent}%`
            : `上传中 ${percent}%`;
          setStatus(
            `上传 ${fileIndex + 1}/${totalFiles}: ${percent}% (${uploadedChunks}/${totalChunks})`,
            false
          );
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
    el.uploadBtn.textContent = totalFiles > 1 ? `合并 ${fileIndex + 1}/${totalFiles}` : "合并中...";
    setStatus(`合并文件 ${fileIndex + 1}/${totalFiles}...`, false);
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

    p.sourceId = data.source_id;
    p.sourceKind = data.kind || "file";
    p.playbackUrl = data.playback_url || "";
    pe.streamMeta.textContent = `来源: ${data.name} (${data.size_mb || "?"}MB)`;
    pe.sourceName.textContent = `— ${data.name}`;
    addLog(`✓ 已加载: ${data.name} (${data.size_mb}MB)`, panelIdx);
    updateButtonStates();
    return data;
  } finally {
    if (watchdog) {
      clearInterval(watchdog);
      watchdog = null;
    }
    if (state.uploadAbortController) {
      try { state.uploadAbortController.abort(); } catch {}
      state.uploadAbortController = null;
    }
  }
}

async function uploadVideo() {
  const actionToken = claimAction("upload");
  const panelIdx = getNextAvailablePanel();
  const files = Array.from(el.videoFile.files || []);

  if (files.length === 0) {
    addLog("⚠ 请先选择视频文件", panelIdx);
    releaseAction(actionToken);
    return;
  }

  const selectedFiles = files.slice(0, PANEL_COUNT);
  const targetPanels = getTargetPanelsForFiles(selectedFiles.length);

  if (files.length > PANEL_COUNT) {
    addLog(
      `⚠ 当前最多同时加载 ${PANEL_COUNT} 个视频，已使用前 ${selectedFiles.length} 个文件`,
      targetPanels[0] ?? 0,
      true
    );
  }

  el.uploadBtn.disabled = true;
  const originalText = el.uploadBtn.textContent;
  let successCount = 0;
  let firstError = null;

  try {
    for (let i = 0; i < targetPanels.length; i++) {
      const targetPanelIdx = targetPanels[i];
      const file = selectedFiles[i];
      const p = state.panels[targetPanelIdx];

      if (p.runId || p.eventSource) {
        await stopAnalysisForPanel(targetPanelIdx, true, true);
      }
      if (!isActionActive(actionToken)) return;

      try {
        await uploadSingleVideoToPanel(
          file,
          targetPanelIdx,
          actionToken,
          i,
          selectedFiles.length
        );
        successCount += 1;
      } catch (err) {
        if (!firstError) firstError = err;
        addLog(`❌ 上传失败: ${file.name} - ${err.message}`, targetPanelIdx, true);
      }
    }

    if (!isActionActive(actionToken)) return;

    if (successCount > 0) {
      setStatus(
        successCount === selectedFiles.length
          ? `${successCount} 个视频已就绪`
          : `已就绪 ${successCount}/${selectedFiles.length} 个视频`,
        true
      );
      el.startBtn.classList.add("pulse-ready");
      setTimeout(() => el.startBtn.classList.remove("pulse-ready"), 2000);
    } else if (firstError) {
      throw firstError;
    }
  } finally {
    el.uploadBtn.disabled = false;
    el.uploadBtn.textContent = originalText;
    releaseAction(actionToken);
  }
}

async function registerUrl() {
  const actionToken = claimAction("connect_url");
  const panelIdx = getNextAvailablePanel();
  const p = state.panels[panelIdx];
  const pe = el.panels[panelIdx];

  if (p.runId || p.eventSource) {
    await stopAnalysisForPanel(panelIdx, true, true);
  }
  if (!isActionActive(actionToken)) return;

  const url = el.streamUrl.value.trim();
  if (!url) {
    addLog("⚠ 请输入流媒体 URL", panelIdx);
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

    p.sourceId = data.source_id;
    p.sourceKind = data.kind || "url";
    p.playbackUrl = data.playback_url || url;
    pe.streamMeta.textContent = `来源: ${url}`;
    pe.sourceName.textContent = `— ${url.length > 30 ? url.slice(0, 30) + "..." : url}`;
    setStatus("流地址已就绪", true);
    addLog(`✓ 已接入: ${url}`, panelIdx);
    updateButtonStates();
    el.startBtn.classList.add("pulse-ready");
    setTimeout(() => el.startBtn.classList.remove("pulse-ready"), 2000);
  } finally {
    el.urlBtn.disabled = false;
    releaseAction(actionToken);
  }
}

async function loadLocalFile() {
  const actionToken = claimAction("load_local");
  const panelIdx = getNextAvailablePanel();
  const p = state.panels[panelIdx];
  const pe = el.panels[panelIdx];

  if (p.runId || p.eventSource) {
    await stopAnalysisForPanel(panelIdx, true, true);
  }
  if (!isActionActive(actionToken)) return;

  const localPath = el.localPath.value.trim();
  if (!localPath) {
    addLog("⚠ 请输入服务器上的文件路径", panelIdx);
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

    p.sourceId = data.source_id;
    p.sourceKind = data.kind || "file";
    p.playbackUrl = data.playback_url || "";
    pe.streamMeta.textContent = `来源: ${data.name} (${data.size_mb || "?"}MB)`;
    pe.sourceName.textContent = `— ${data.name}`;
    setStatus("视频已就绪", true);
    addLog(`✓ 已加载服务器文件: ${data.name}`, panelIdx);
    updateButtonStates();
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
  updateSelectedFilesLabel([]);
  setUploadFeedback("", false);

  for (let i = 0; i < PANEL_COUNT; i++) {
    addLog("欢迎使用 Video Intelligence Studio。请先接入视频源，然后点击「开始分析」。", i);
  }
}

/* ── Event Listeners ── */
el.sourceTypeSeg.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-source-type]");
  if (!btn) return;
  if (state.panels.some(p => p.runId)) {
    await stopAnalysis(true, false, true, true);
  }
  state.sourceType = btn.dataset.sourceType;
  updateSourceTypeUI();
});

el.modeSeg.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (!btn) return;

  if (state.panels.some(p => p.runId)) {
    await stopAnalysis(true, false, true, true);
  }
  state.mode = btn.dataset.mode;
  updateModeUI();
  for (let i = 0; i < PANEL_COUNT; i++) {
    clearLogs(i);
  }
  updateButtonStates();
});

el.uploadBtn.addEventListener("click", async () => {
  try { await uploadVideo(); }
  catch (err) { addLog(`❌ 上传失败: ${err.message}`, 0, true); setStatus("上传失败", false); }
});

el.urlBtn.addEventListener("click", async () => {
  try { await registerUrl(); }
  catch (err) { addLog(`❌ 连接失败: ${err.message}`, 0, true); setStatus("连接失败", false); }
});

el.localBtn.addEventListener("click", async () => {
  try { await loadLocalFile(); }
  catch (err) { addLog(`❌ 加载失败: ${err.message}`, 0, true); setStatus("加载失败", false); }
});

el.startBtn.addEventListener("click", async () => {
  const origText = el.startBtn.textContent;
  el.startBtn.textContent = "启动中...";
  el.startBtn.disabled = true;
  try {
    await startAllStreams();
  } catch (err) {
    addLog(`❌ 启动失败: ${err.message}`, 0, true);
    setStatus("启动失败", false);
  } finally {
    el.startBtn.textContent = origText;
    updateButtonStates();
  }
});

el.stopBtn.addEventListener("click", async () => {
  await stopAnalysis(true, false);
  for (let i = 0; i < PANEL_COUNT; i++) {
    if (state.panels[i].sourceId) {
      addLog("■ 分析已停止，可重新点击「开始分析」继续", i);
    }
  }
});

el.applyPromptBtn.addEventListener("click", async () => {
  if (state.mode !== "infer") return;
  if (!state.panels.some(p => p.sourceId)) {
    addLog("⚠ 请先接入视频源", 0, true);
    return;
  }
  for (let i = 0; i < PANEL_COUNT; i++) {
    if (state.panels[i].sourceId) {
      addLog("→ 已应用新 Prompt，重新开始流式推理...", i, true);
    }
  }
  await startAllStreams();
});

el.resetPromptBtn.addEventListener("click", () => {
  el.promptInput.value = state.defaultPrompt;
  for (let i = 0; i < PANEL_COUNT; i++) {
    if (state.panels[i].sourceId) {
      addLog("→ Prompt 已恢复默认", i, true);
    }
  }
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
    updateSelectedFilesLabel(Array.from(e.dataTransfer.files));
  }
});
el.videoFile.addEventListener("change", () => {
  updateSelectedFilesLabel(Array.from(el.videoFile.files || []));
});

window.addEventListener("beforeunload", () => {
  for (let i = 0; i < PANEL_COUNT; i++) {
    const p = state.panels[i];
    if (p.eventSource) p.eventSource.close();
    stopNativePlayback(i);
    if (p.sourceId && p.runId) {
      try {
        navigator.sendBeacon(
          "/api/control/stop",
          new Blob(
            [JSON.stringify({ source_id: p.sourceId, run_id: p.runId })],
            { type: "application/json" }
          )
        );
      } catch {
        // ignore
      }
    }
  }
});

boot().catch((err) => {
  addLog(`❌ 初始化失败: ${err.message}`, 0, true);
  setStatus("初始化失败", false);
});
