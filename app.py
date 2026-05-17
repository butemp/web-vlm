import asyncio
import concurrent.futures
import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from PIL import Image

logger = logging.getLogger("web_vlm")
_LOG_LEVEL_STR = os.getenv("WEB_VLM_LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_STR, logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=_LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
logger.setLevel(_LOG_LEVEL)

try:
    import torch
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        StoppingCriteria,
        StoppingCriteriaList,
    )
    from ultralytics import YOLO
except Exception:
    torch = None
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
    StoppingCriteria = None
    StoppingCriteriaList = None
    YOLO = None

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULT_PROMPT = "请简单描述一下这个视频"
SUMMARY_PROMPT = "请用一两句话简单总结当前摄像头画面的主要内容。"
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
MAX_CONCURRENT_RUNS = 4

# ---------------------------------------------------------------------------
# Hardcoded model paths (edit here before deploying).
# ---------------------------------------------------------------------------
# QWEN_MODEL_PATH = "/data/models/Qwen2.5-VL-3B-Instruct"
# YOLO_MODEL_PATH = "/data/models/yolov8x.pt"
QWEN_MODEL_PATH = "/home/nanguoshun/ckpt/Qwen/Qwen2.5-VL-3B-Instruct"
YOLO_MODEL_PATH = "/home/nanguoshun/ckpt/yolov8x.pt"

GPU_DEVICE = "cuda:0"
QWEN_MAX_NEW_TOKENS = 256
QWEN_MAX_IMAGE_EDGE = 640
YOLO_CONF = 0.25
YOLO_IMGSZ = 640
YOLO_STREAM_IMGSZ = 416
YOLO_LIVE_STREAM_IMGSZ = 352
YOLO_DRAW_EVERY_N_FRAMES = 3
YOLO_STREAM_INFER_INTERVAL_SEC = 0.45
YOLO_LIVE_STREAM_INFER_INTERVAL_SEC = 0.75
STREAM_MAX_EDGE = 720
STREAM_JPEG_QUALITY = 65
STREAM_TARGET_FPS = 25
LIVE_STREAM_MAX_EDGE = 540
LIVE_STREAM_JPEG_QUALITY = 55
LIVE_STREAM_TARGET_FPS = 20
DETECT_STREAM_MAX_EDGE = 540
DETECT_STREAM_TARGET_FPS = 20
LIVE_DETECT_STREAM_MAX_EDGE = 512
LIVE_DETECT_STREAM_TARGET_FPS = 10
LIVE_DETECT_JPEG_QUALITY = 52
LIVE_DETECT_MIN_FRAME_STEP = 1
NETWORK_OPEN_TIMEOUT_MSEC = 15000
NETWORK_READ_TIMEOUT_MSEC = 8000
INFER_CACHE_EVERY_N_FRAMES = 2
INFER_MIN_INTERVAL_SEC = 1.2
INFER_CACHE_MAX_AGE_SEC = 8.0
INFER_CACHE_WAIT_SEC = 5.0
INFER_CACHE_POLL_INTERVAL_SEC = 0.05
PRELOAD_QWEN_ON_STARTUP = True

# Runtime model cache
_qwen_lock = threading.Lock()
_yolo_lock = threading.Lock()
_qwen_model = None
_qwen_processor = None
_yolo_model = None

# Shared detection cache to avoid duplicate YOLO inference between video stream
# and detect SSE stream.
_detect_cache_lock = threading.Lock()
_detect_cache: Dict[str, Dict[str, object]] = {}
_infer_frame_cache_lock = threading.Lock()
_infer_frame_cache: Dict[str, Dict[str, object]] = {}

_active_runs_lock = threading.Lock()
_active_runs: Dict[str, str] = {}
_ffmpeg_env_lock = threading.Lock()

# Separate executor for JPEG encoding so it doesn't compete with YOLO/Qwen inference
_encode_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="jpeg-enc"
)
# Separate executor for YOLO detection to avoid starving the default asyncio pool
_yolo_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="yolo-det"
)

app = FastAPI(title="Realtime Video Inference Demo", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def no_cache_static_middleware(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") and path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)

    req_id = uuid.uuid4().hex[:8]
    method = request.method
    client = request.client.host if request.client else "-"
    start = time.monotonic()
    logger.info("[REQ %s] -> %s %s client=%s", req_id, method, path, client)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "[REQ %s] !! %s %s cost=%.1fms",
            req_id,
            method,
            path,
            (time.monotonic() - start) * 1000.0,
        )
        raise

    logger.info(
        "[REQ %s] <- %s %s status=%d cost=%.1fms",
        req_id,
        method,
        path,
        response.status_code,
        (time.monotonic() - start) * 1000.0,
    )
    return response


@app.on_event("startup")
async def _startup_preload_models() -> None:
    if not PRELOAD_QWEN_ON_STARTUP:
        return

    logger.info("服务启动：开始预加载 VLM-Online 模型")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _ensure_qwen_loaded)
        logger.info("服务启动：VLM-Online 模型预加载完成")
    except Exception as exc:
        logger.exception("服务启动：VLM-Online 模型预加载失败: %s", exc)
        raise RuntimeError(f"VLM-Online 模型预加载失败: {exc}") from exc

# ---------------------------------------------------------------------------
# In-memory registry for uploaded files / online stream URLs.
# ---------------------------------------------------------------------------
SOURCES: Dict[str, Dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _open_capture_value(value: str) -> tuple[cv2.VideoCapture, float, bool]:
    ffmpeg_options = _build_ffmpeg_capture_options(value)

    with _ffmpeg_env_lock:
        if ffmpeg_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_options
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

        open_t0 = time.monotonic()
        cap = cv2.VideoCapture(value, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(value, cv2.CAP_ANY)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频源: {value}")
    # Optimize capture settings for smoother playback
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        # For network streams, set longer timeouts
        if value.startswith(("http://", "https://", "rtsp://")):
            # Network sources benefit from a slightly deeper decode buffer.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 8)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, NETWORK_OPEN_TIMEOUT_MSEC)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, NETWORK_READ_TIMEOUT_MSEC)
    except Exception:
        cap.release()
        raise
    return cap, time.monotonic() - open_t0, bool(ffmpeg_options)


def _open_capture(source_id: str) -> cv2.VideoCapture:
    source = SOURCES.get(source_id)
    if not source:
        raise ValueError("source_id not found")
    value = source["value"]

    cap, open_cost, has_ffmpeg_options = _open_capture_value(value)
    logger.info(
        "视频源打开成功 source=%s kind=%s open_cost=%.2fs ffmpeg_opts=%s",
        _short_id(source_id),
        source.get("kind", "unknown"),
        open_cost,
        "on" if has_ffmpeg_options else "off",
    )
    return cap


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    if not name:
        name = "video.mp4"
    return name


def _start_run(source_id: str) -> tuple[str, List[str]]:
    run_id = str(uuid.uuid4())
    with _active_runs_lock:
        replaced_sources: List[str] = []
        # If the same source already has a run, replace only that run
        if source_id in _active_runs:
            pass  # will be overwritten below
        elif len(_active_runs) >= MAX_CONCURRENT_RUNS:
            # Evict the oldest (first key in dict) to make room
            oldest_sid = next(iter(_active_runs))
            replaced_sources.append(oldest_sid)
            del _active_runs[oldest_sid]
        _active_runs[source_id] = run_id
        active_count = len(_active_runs)
    logger.info(
        "会话启动 run=%s source=%s active_runs=%d replaced=%d",
        _short_id(run_id),
        _short_id(source_id),
        active_count,
        len(replaced_sources),
    )
    return run_id, replaced_sources


def _is_run_active(source_id: str, run_id: str) -> bool:
    with _active_runs_lock:
        return _active_runs.get(source_id) == run_id


def _get_active_run_id(source_id: str) -> str:
    with _active_runs_lock:
        return _active_runs.get(source_id, "")


def _stop_all_runs(reason: str = "") -> List[str]:
    with _active_runs_lock:
        stopped_sources = list(_active_runs.keys())
        _active_runs.clear()
    if stopped_sources:
        logger.info(
            "全量停止会话 count=%d reason=%s sources=%s",
            len(stopped_sources),
            reason or "-",
            ",".join([_short_id(x) for x in stopped_sources]),
        )
    return stopped_sources


def _stop_run(source_id: str, run_id: str = "") -> bool:
    with _active_runs_lock:
        current = _active_runs.get(source_id)
        if current is None:
            logger.info(
                "会话停止忽略 source=%s reason=no_active_run",
                _short_id(source_id),
            )
            return False
        if run_id and current != run_id:
            logger.warning(
                "会话停止忽略 source=%s requested_run=%s current_run=%s",
                _short_id(source_id),
                _short_id(run_id),
                _short_id(current),
            )
            return False
        _active_runs.pop(source_id, None)
        active_count = len(_active_runs)
    logger.info(
        "会话停止成功 source=%s run=%s active_runs=%d",
        _short_id(source_id),
        _short_id(current),
        active_count,
    )
    return True


def _clear_detect_cache(source_id: str) -> None:
    with _detect_cache_lock:
        _detect_cache.pop(source_id, None)


def _clear_infer_cache(source_id: str) -> None:
    with _infer_frame_cache_lock:
        _infer_frame_cache.pop(source_id, None)


def _get_latest_infer_cache_frame(
    source_id: str,
    run_id: str,
    last_frame_idx: int,
) -> tuple[int, np.ndarray] | None:
    with _infer_frame_cache_lock:
        state = _infer_frame_cache.get(source_id)
        if not state:
            return None
        if str(state.get("run_id", "")) != run_id:
            return None
        frame_idx = int(state.get("frame", -1))
        if frame_idx <= last_frame_idx:
            return None
        ts = float(state.get("ts", 0.0))
        if (time.time() - ts) > INFER_CACHE_MAX_AGE_SEC:
            return None
        image = state.get("image")
        if image is None:
            return None
        return frame_idx, image.copy()


def _build_ffmpeg_capture_options(url: str) -> str:
    lower = url.lower()
    if lower.startswith("rtsp://"):
        # Prefer transport stability over ultra-low latency for long-running sessions.
        return (
            "rtsp_transport;tcp|stimeout;15000000|rw_timeout;15000000|"
            "fflags;genpts|flags;low_delay"
        )
    if ".m3u8" in lower:
        # HLS stream: keep reconnect capability, avoid aggressive no-buffer mode
        # which can amplify packet jitter into decode corruption on weak links.
        return (
            "fflags;genpts|flags;low_delay|"
            "analyzeduration;2000000|probesize;2000000|"
            "reconnect;1|reconnect_streamed;1|reconnect_delay_max;2|"
            "rw_timeout;15000000|timeout;10000000"
        )
    return ""


def _short_id(value: str, keep: int = 8) -> str:
    if not value:
        return "-"
    return value[:keep]


def _source_brief(source_id: str) -> str:
    source = SOURCES.get(source_id)
    if not source:
        return f"{_short_id(source_id)}(missing)"
    kind = str(source.get("kind", "unknown"))
    name = str(source.get("name", ""))
    value = str(source.get("value", ""))
    if kind == "url":
        v = value if len(value) <= 110 else (value[:110] + "...")
    else:
        v = Path(value).name
    return f"{_short_id(source_id)} kind={kind} name={name} value={v}"


def _require_runtime_deps() -> None:
    if (
        torch is None
        or AutoProcessor is None
        or YOLO is None
        or StoppingCriteria is None
        or StoppingCriteriaList is None
    ):
        raise RuntimeError(
            "缺少推理依赖，请安装 torch/transformers/ultralytics/Pillow 后重启服务。"
        )


def _require_gpu() -> None:
    if torch is None or (not torch.cuda.is_available()):
        raise RuntimeError("未检测到 CUDA GPU，当前配置要求使用 GPU 推理。")


def _ensure_qwen_loaded():
    global _qwen_model, _qwen_processor
    if _qwen_model is not None and _qwen_processor is not None:
        return _qwen_model, _qwen_processor

    with _qwen_lock:
        if _qwen_model is not None and _qwen_processor is not None:
            return _qwen_model, _qwen_processor

        _require_runtime_deps()
        _require_gpu()

        if not os.path.exists(QWEN_MODEL_PATH):
            raise RuntimeError(f"VLM-Online 模型路径不存在: {QWEN_MODEL_PATH}")

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("开始加载 VLM-Online 模型: %s", QWEN_MODEL_PATH)
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto",
        )
        _qwen_model.eval()
        _qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
        logger.info("VLM-Online 模型加载完成")

    return _qwen_model, _qwen_processor


def _ensure_yolo_loaded():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model

        _require_runtime_deps()
        _require_gpu()

        if not os.path.exists(YOLO_MODEL_PATH):
            raise RuntimeError(f"VLM-Detect 模型路径不存在: {YOLO_MODEL_PATH}")

        logger.info("开始加载 VLM-Detect 模型: %s", YOLO_MODEL_PATH)
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info("VLM-Detect 模型加载完成")

    return _yolo_model


def _resize_by_max_edge(frame: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return frame

    scale = max_edge / float(longest)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _resize_for_vlm(frame: np.ndarray, max_edge: int = QWEN_MAX_IMAGE_EDGE) -> np.ndarray:
    return _resize_by_max_edge(frame, max_edge=max_edge)


class ThreadedFrameReader:
    """Background thread video reader with frame queue for smooth playback."""

    STREAM_CONSECUTIVE_FAIL_LIMIT = 50
    STREAM_MAX_RECONNECTS = 5
    STREAM_RECONNECT_BACKOFF_SEC = 1.0

    def __init__(
        self,
        cap: cv2.VideoCapture,
        queue_size: int = 8,
        skip_frames: int = 0,
        is_file: bool = False,
        drop_if_full: bool = True,
        source_id: str = "",
    ):
        self.cap = cap
        self.skip_frames = skip_frames
        self.is_file = is_file
        self.drop_if_full = drop_if_full
        self.source_id = source_id
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _try_reconnect(self) -> bool:
        """Attempt to reopen the capture for network streams."""
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            new_cap = _open_capture(self.source_id)
            self.cap = new_cap
            logger.info(
                "读帧线程重连成功 source=%s", _short_id(self.source_id)
            )
            return True
        except Exception as exc:
            logger.warning(
                "读帧线程重连失败 source=%s err=%s",
                _short_id(self.source_id),
                exc,
            )
            return False

    def _reader_loop(self):
        retry_count = 0
        max_retries = 10
        stream_consecutive_fails = 0
        stream_reconnects = 0
        while not self.stopped:
            if self.queue.full():
                if self.drop_if_full:
                    # Live streams prefer low latency: drain all stale frames
                    # so downstream always sees the most recent frame.
                    while not self.queue.empty():
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            break
                else:
                    # File playback prefers stable speed: apply backpressure and do not drop.
                    time.sleep(0.002)
                    continue

            # Skip frames if needed
            for _ in range(self.skip_frames):
                if self.stopped:
                    return
                if not self.cap.grab():
                    break

            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.stopped = True
                        try:
                            self.queue.put_nowait((False, None, -1.0))
                        except queue.Full:
                            pass
                        return
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.15)
                    continue
                else:
                    # Network stream: track consecutive failures and attempt reconnect
                    stream_consecutive_fails += 1
                    if stream_consecutive_fails >= self.STREAM_CONSECUTIVE_FAIL_LIMIT:
                        if stream_reconnects >= self.STREAM_MAX_RECONNECTS:
                            logger.warning(
                                "读帧线程放弃重连 source=%s reconnects=%d",
                                _short_id(self.source_id),
                                stream_reconnects,
                            )
                            self.stopped = True
                            try:
                                self.queue.put_nowait((False, None, -1.0))
                            except queue.Full:
                                pass
                            return
                        backoff = self.STREAM_RECONNECT_BACKOFF_SEC * (
                            stream_reconnects + 1
                        )
                        logger.info(
                            "读帧线程连续失败 source=%s fails=%d, %.1fs后尝试重连(%d/%d)",
                            _short_id(self.source_id),
                            stream_consecutive_fails,
                            backoff,
                            stream_reconnects + 1,
                            self.STREAM_MAX_RECONNECTS,
                        )
                        time.sleep(backoff)
                        if self.stopped:
                            return
                        if self._try_reconnect():
                            stream_consecutive_fails = 0
                            stream_reconnects += 1
                        else:
                            stream_reconnects += 1
                    else:
                        time.sleep(0.05)
                    continue

            retry_count = 0
            stream_consecutive_fails = 0
            pos_msec = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            if np.isnan(pos_msec) or pos_msec < 0:
                pos_msec = -1.0
            try:
                self.queue.put((True, frame, pos_msec), timeout=0.1)
            except queue.Full:
                pass

    def read(self, timeout: float = 0.015) -> tuple[bool, Optional[np.ndarray], float]:
        if self.stopped and self.queue.empty():
            return False, None, -1.0
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return False, None, -1.0

    def stop(self):
        self.stopped = True
        # Join the reader thread FIRST so it exits cap.read() before we release.
        if self.thread.is_alive():
            join_timeout = max(
                0.8,
                (float(NETWORK_READ_TIMEOUT_MSEC) / 1000.0) + 0.6,
            )
            self.thread.join(timeout=join_timeout)
            if self.thread.is_alive():
                logger.warning(
                    "读帧线程未及时退出，后续将由后台超时回收 thread=%s",
                    self.thread.name,
                )
        try:
            self.cap.release()
        except Exception:
            pass


def _grab_and_read_frame(cap: cv2.VideoCapture, n_skip: int) -> tuple[bool, np.ndarray]:
    if n_skip > 0:
        for _ in range(n_skip):
            if not cap.grab():
                return False, None
    return cap.read()


def _resize_draw_encode_jpeg(
    frame: np.ndarray,
    max_edge: int,
    jpeg_quality: int,
    detections: List[Dict[str, object]] | None = None,
) -> tuple[np.ndarray, bytes | None]:
    display_frame = _resize_by_max_edge(frame, max_edge=max_edge)
    if detections:
        display_frame = _draw_detections(display_frame, detections)

    ok, encoded = cv2.imencode(
        ".jpg",
        display_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not ok:
        return display_frame, None
    return display_frame, encoded.tobytes()


_RunStopBase = StoppingCriteria if StoppingCriteria is not None else object


class _RunStopCriteria(_RunStopBase):
    def __init__(self, source_id: str, run_id: str):
        self.source_id = source_id
        self.run_id = run_id

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return not _is_run_active(self.source_id, self.run_id)


def _qwen_caption(
    frame: np.ndarray,
    prompt: str,
    source_id: str = "",
    run_id: str = "",
) -> str:
    model, processor = _ensure_qwen_loaded()
    frame = _resize_for_vlm(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    with _qwen_lock:
        model_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_inputs = model_inputs.to(model.device)

        generate_kwargs = {
            "max_new_tokens": QWEN_MAX_NEW_TOKENS,
            "do_sample": False,
        }
        if source_id and run_id:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [_RunStopCriteria(source_id, run_id)]
            )

        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, **generate_kwargs)

    input_ids = model_inputs["input_ids"]
    generated_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text or "未生成有效文本。"


def _capture_summary_frame(addr: str) -> np.ndarray:
    cap = None
    try:
        cap, open_cost, has_ffmpeg_options = _open_capture_value(addr)
        logger.info(
            "summary 视频源打开成功 open_cost=%.2fs ffmpeg_opts=%s addr=%s",
            open_cost,
            "on" if has_ffmpeg_options else "off",
            addr,
        )

        # Drain a few frames from buffered live sources so the summary is closer
        # to the current camera view without keeping the HTTP caller waiting long.
        for _ in range(3):
            try:
                if not cap.grab():
                    break
            except Exception:
                break

        last_frame = None
        for _ in range(8):
            ret, frame = cap.read()
            if ret and frame is not None:
                last_frame = frame
                break
            time.sleep(0.03)

        if last_frame is None:
            raise ValueError("无法从视频源读取有效画面")
        return last_frame
    finally:
        if cap is not None:
            cap.release()


def _extract_yolo_name(names: object, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _yolo_detect(
    frame: np.ndarray,
    targets: List[str],
    imgsz: int = YOLO_IMGSZ,
) -> List[Dict[str, object]]:
    model = _ensure_yolo_loaded()
    target_set = {x.lower().strip() for x in targets if x.strip()}
    frame_h, frame_w = frame.shape[:2]
    safe_w = max(1, frame_w)
    safe_h = max(1, frame_h)

    with _yolo_lock:
        results = model.predict(
            source=frame,
            conf=YOLO_CONF,
            imgsz=imgsz,
            device=GPU_DEVICE,
            verbose=False,
        )

    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    names = result.names
    if boxes is None:
        return []

    detections: List[Dict[str, object]] = []
    for box in boxes:
        cls_id = int(box.cls.item())
        label = _extract_yolo_name(names, cls_id)
        if target_set and label.lower() not in target_set:
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "label": label,
                "conf": conf,
                "xyxy": [x1, y1, x2, y2],
                # Keep normalized coords so boxes stay aligned across
                # different render resolutions.
                "norm_xyxy": [
                    float(x1) / safe_w,
                    float(y1) / safe_h,
                    float(x2) / safe_w,
                    float(y2) / safe_h,
                ],
            }
        )

    detections.sort(key=lambda x: x["conf"], reverse=True)
    return detections


def _draw_detections(frame: np.ndarray, detections: List[Dict[str, object]]) -> np.ndarray:
    color_palette = [
        (80, 220, 255),
        (255, 172, 90),
        (128, 255, 128),
        (255, 128, 212),
        (180, 180, 255),
    ]
    fh, fw = frame.shape[:2]

    for i, det in enumerate(detections):
        norm_xyxy = det.get("norm_xyxy")
        if (
            isinstance(norm_xyxy, (list, tuple))
            and len(norm_xyxy) == 4
        ):
            nx1, ny1, nx2, ny2 = [float(v) for v in norm_xyxy]
            x1 = int(nx1 * fw)
            y1 = int(ny1 * fh)
            x2 = int(nx2 * fw)
            y2 = int(ny2 * fh)
        else:
            x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]

        x1 = max(0, min(fw - 1, x1))
        y1 = max(0, min(fh - 1, y1))
        x2 = max(0, min(fw - 1, x2))
        y2 = max(0, min(fh - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        label = det["label"]
        conf = det["conf"]
        color = color_palette[i % len(color_palette)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    return frame


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    # Inject a dynamic cache-busting timestamp so browsers never use stale JS/CSS
    _ts = str(int(time.time()))
    html = html.replace("__CACHE_BUST__", _ts)
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/defaults")
async def defaults() -> JSONResponse:
    return JSONResponse({"default_prompt": DEFAULT_PROMPT})


@app.get("/summary")
async def summary(
    addr: str = Query(..., description="摄像头或视频流地址，如 rtsp/http/hls"),
    prompt: str = Query(SUMMARY_PROMPT, description="可选的单帧总结指令"),
) -> JSONResponse:
    """Capture one frame from a stream URL and return a VLM summary."""

    addr = (addr or "").strip()
    prompt = (prompt or SUMMARY_PROMPT).strip() or SUMMARY_PROMPT
    if not addr:
        raise HTTPException(status_code=400, detail="addr 不能为空")

    request_t0 = time.monotonic()
    loop = asyncio.get_event_loop()
    try:
        frame = await loop.run_in_executor(None, _capture_summary_frame, addr)
    except ValueError as exc:
        logger.warning("summary 抓帧失败 addr=%s err=%s", addr, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("summary 打开或读取视频源异常 addr=%s", addr)
        raise HTTPException(status_code=500, detail=f"读取视频源失败: {exc}") from exc

    try:
        text = await loop.run_in_executor(None, _qwen_caption, frame, prompt)
    except Exception as exc:
        logger.exception("summary VLM 推理失败 addr=%s", addr)
        raise HTTPException(status_code=500, detail=f"VLM 推理失败: {exc}") from exc

    cost_ms = (time.monotonic() - request_t0) * 1000.0
    logger.info(
        "summary 完成 cost=%.1fms addr=%s prompt_len=%d",
        cost_ms,
        addr,
        len(prompt),
    )
    return JSONResponse(
        {
            "summary": text,
            "cost_ms": round(cost_ms, 1),
        }
    )


# ---------------------------------------------------------------------------
# Chunked Upload — reliable through tunnels / slow connections
# ---------------------------------------------------------------------------
_upload_sessions: Dict[str, Dict[str, object]] = {}
_upload_sessions_lock = threading.Lock()
_UPLOAD_SESSION_TTL_SEC = 1800  # 30 minutes


def _cleanup_stale_uploads() -> None:
    """Remove upload sessions older than TTL and their chunk directories."""
    now = time.time()
    to_remove: list[str] = []
    with _upload_sessions_lock:
        for uid, sess in _upload_sessions.items():
            if now - float(sess.get("created", 0)) > _UPLOAD_SESSION_TTL_SEC:
                to_remove.append(uid)
        for uid in to_remove:
            sess = _upload_sessions.pop(uid, None)
            if sess:
                chunk_dir = sess.get("chunk_dir", "")
                if chunk_dir:
                    shutil.rmtree(str(chunk_dir), ignore_errors=True)
    if to_remove:
        logger.info("清理过期上传会话: %d 个", len(to_remove))


async def _upload_cleanup_task() -> None:
    """Periodic background task to clean up stale upload sessions."""
    while True:
        await asyncio.sleep(300)  # every 5 minutes
        try:
            _cleanup_stale_uploads()
        except Exception:
            logger.exception("上传清理任务异常")


@app.on_event("startup")
async def _start_upload_cleanup() -> None:
    asyncio.create_task(_upload_cleanup_task())


@app.post("/api/upload/init")
async def upload_init(payload: Dict[str, object]) -> JSONResponse:
    """Initialize a chunked upload session."""

    filename = _safe_filename(str(payload.get("filename", "video.mp4")))
    total_size = int(payload.get("total_size", 0))
    total_chunks = int(payload.get("total_chunks", 0))

    if total_size <= 0 or total_chunks <= 0:
        raise HTTPException(status_code=400, detail="参数无效")
    if total_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE // (1024*1024)} MB",
        )

    upload_id = str(uuid.uuid4())
    chunk_dir = UPLOAD_DIR / f"chunks_{upload_id}"
    chunk_dir.mkdir(exist_ok=True)

    with _upload_sessions_lock:
        _upload_sessions[upload_id] = {
            "filename": filename,
            "total_size": total_size,
            "total_chunks": total_chunks,
            "received": set(),
            "chunk_dir": str(chunk_dir),
            "created": time.time(),
        }
    logger.info(
        "上传初始化 upload=%s file=%s size_mb=%.2f total_chunks=%d",
        _short_id(upload_id),
        filename,
        total_size / (1024 * 1024),
        total_chunks,
    )

    return JSONResponse({"upload_id": upload_id, "message": "上传会话已创建"})


@app.get("/api/upload/init")
async def upload_init_get(
    filename: str = Query(""),
    total_size: int = Query(0),
    total_chunks: int = Query(0),
) -> JSONResponse:
    # Backward/edge compatibility for clients that accidentally send GET.
    if filename and total_size > 0 and total_chunks > 0:
        return await upload_init(
            {
                "filename": filename,
                "total_size": total_size,
                "total_chunks": total_chunks,
            }
        )
    return JSONResponse(
        {
            "ok": False,
            "message": "请使用 POST /api/upload/init，或在 GET 中提供 filename/total_size/total_chunks",
        }
    )


@app.get("/api/upload/init/")
async def upload_init_get_slash(
    filename: str = Query(""),
    total_size: int = Query(0),
    total_chunks: int = Query(0),
) -> JSONResponse:
    return await upload_init_get(
        filename=filename,
        total_size=total_size,
        total_chunks=total_chunks,
    )


@app.post("/api/upload/chunk")
async def upload_chunk(
    upload_id: str = Query(...),
    chunk_index: int = Query(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    """Receive a single chunk."""
    with _upload_sessions_lock:
        session = _upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(status_code=404, detail="上传会话不存在或已过期")

    if chunk_index < 0 or chunk_index >= session["total_chunks"]:
        raise HTTPException(status_code=400, detail="分片索引无效")

    chunk_path = Path(session["chunk_dir"]) / f"{chunk_index:06d}"
    data = await file.read()
    chunk_path.write_bytes(data)

    with _upload_sessions_lock:
        session["received"].add(chunk_index)
        received_count = len(session["received"])
        total_chunks = int(session["total_chunks"])

    if chunk_index == 0 or received_count == total_chunks or (received_count % 50 == 0):
        logger.info(
            "上传分片 upload=%s progress=%d/%d chunk=%d bytes=%d",
            _short_id(upload_id),
            received_count,
            total_chunks,
            chunk_index,
            len(data),
        )

    return JSONResponse({
        "chunk_index": chunk_index,
        "received": received_count,
        "total_chunks": total_chunks,
    })


@app.post("/api/upload/complete")
async def upload_complete(payload: Dict[str, str]) -> JSONResponse:
    """Merge all chunks into the final file."""
    upload_id = (payload.get("upload_id") or "").strip()

    with _upload_sessions_lock:
        session = _upload_sessions.get(upload_id)
    if not session:
        raise HTTPException(status_code=404, detail="上传会话不存在")

    total_chunks = session["total_chunks"]
    if len(session["received"]) < total_chunks:
        missing = total_chunks - len(session["received"])
        raise HTTPException(status_code=400, detail=f"还有 {missing} 个分片未上传")

    filename = session["filename"]
    source_id = str(uuid.uuid4())
    target_path = UPLOAD_DIR / f"{source_id}_{filename}"
    chunk_dir = Path(session["chunk_dir"])

    # Merge chunks
    total_written = 0
    with target_path.open("wb") as fp:
        for i in range(total_chunks):
            cp = chunk_dir / f"{i:06d}"
            chunk_data = cp.read_bytes()
            fp.write(chunk_data)
            total_written += len(chunk_data)

    # Clean up chunks
    shutil.rmtree(str(chunk_dir), ignore_errors=True)
    with _upload_sessions_lock:
        _upload_sessions.pop(upload_id, None)

    SOURCES[source_id] = {"kind": "file", "value": str(target_path), "name": filename}
    logger.info(
        "上传完成 upload=%s source=%s file=%s size_mb=%.2f",
        _short_id(upload_id),
        _short_id(source_id),
        filename,
        total_written / (1024 * 1024),
    )
    return JSONResponse(
        {
            "source_id": source_id,
            "kind": "file",
            "name": filename,
            "playback_url": f"/api/source/{source_id}/file",
            "size_mb": round(total_written / (1024 * 1024), 2),
            "message": "视频上传成功",
        }
    )


# Legacy single-file upload (for small files / direct access)
@app.post("/api/source/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:

    filename = _safe_filename(file.filename or "video.mp4")
    source_id = str(uuid.uuid4())
    target_path = UPLOAD_DIR / f"{source_id}_{filename}"

    total = 0
    chunk_size = 4 * 1024 * 1024
    with target_path.open("wb") as fp:
        while chunk := await file.read(chunk_size):
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                fp.close()
                target_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE // (1024*1024)} MB",
                )
            fp.write(chunk)

    if total == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传文件为空")

    SOURCES[source_id] = {"kind": "file", "value": str(target_path), "name": filename}
    logger.info(
        "单文件上传完成 source=%s file=%s size_mb=%.2f",
        _short_id(source_id),
        filename,
        total / (1024 * 1024),
    )
    return JSONResponse(
        {
            "source_id": source_id,
            "kind": "file",
            "name": filename,
            "playback_url": f"/api/source/{source_id}/file",
            "size_mb": round(total / (1024 * 1024), 2),
            "message": "视频上传成功",
        }
    )


@app.post("/api/source/url")
async def register_url(payload: Dict[str, str]) -> JSONResponse:
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL 不能为空")

    source_id = str(uuid.uuid4())
    SOURCES[source_id] = {"kind": "url", "value": url, "name": "online_stream"}
    logger.info("URL 注册成功 source=%s url=%s", _short_id(source_id), url)
    return JSONResponse(
        {
            "source_id": source_id,
            "kind": "url",
            "name": url,
            "playback_url": url,
            "message": "视频流地址已注册",
        }
    )


@app.post("/api/source/local")
async def register_local(payload: Dict[str, str]) -> JSONResponse:
    """Load a video file that already exists on the server."""
    file_path = (payload.get("path") or "").strip()
    if not file_path:
        raise HTTPException(status_code=400, detail="路径不能为空")

    if not os.path.isabs(file_path):
        raise HTTPException(status_code=400, detail="请使用绝对路径，例如 /data/videos/demo.mp4")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    filename = Path(file_path).name
    file_size = os.path.getsize(file_path)
    source_id = str(uuid.uuid4())
    SOURCES[source_id] = {"kind": "file", "value": file_path, "name": filename}
    logger.info(
        "本地文件注册成功 source=%s file=%s size_mb=%.2f",
        _short_id(source_id),
        file_path,
        file_size / (1024 * 1024),
    )
    return JSONResponse(
        {
            "source_id": source_id,
            "kind": "file",
            "name": filename,
            "playback_url": f"/api/source/{source_id}/file",
            "size_mb": round(file_size / (1024 * 1024), 2),
            "message": "服务器本地文件加载成功",
        }
    )


@app.get("/api/source/{source_id}/file")
async def source_file(source_id: str, request: Request):
    source = SOURCES.get(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="source_id 不存在")
    if source.get("kind") != "file":
        raise HTTPException(status_code=400, detail="该 source 不是离线文件")

    path = source.get("value", "")
    if not path or (not os.path.exists(path)):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # Determine media type from extension
    ext = Path(path).suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".m4v": "video/x-m4v",
    }
    media_type = media_types.get(ext, "video/mp4")
    
    return FileResponse(
        path,
        media_type=media_type,
        filename=source.get("name", "video.mp4"),
    )


@app.post("/api/control/start")
async def control_start(payload: Dict[str, str]) -> JSONResponse:
    source_id = (payload.get("source_id") or "").strip()
    if not source_id or source_id not in SOURCES:
        raise HTTPException(status_code=404, detail="source_id 不存在")

    run_id, replaced_sources = _start_run(source_id)
    for sid in replaced_sources:
        _clear_detect_cache(sid)
        _clear_infer_cache(sid)
    _clear_detect_cache(source_id)
    _clear_infer_cache(source_id)
    logger.info(
        "控制启动 source=%s run=%s replaced_sources=%d detail=%s",
        _short_id(source_id),
        _short_id(run_id),
        len(replaced_sources),
        _source_brief(source_id),
    )
    return JSONResponse(
        {
            "source_id": source_id,
            "run_id": run_id,
            "replaced_sources": replaced_sources,
            "message": "分析会话已启动",
        }
    )


@app.get("/api/control/start")
async def control_start_get(source_id: str = Query("")) -> JSONResponse:
    if not source_id:
        return JSONResponse(
            {
                "ok": False,
                "message": "请提供 source_id，或改用 POST /api/control/start",
                "available_sources": list(SOURCES.keys())[:12],
            }
        )
    return await control_start({"source_id": source_id})


@app.get("/api/control/start/")
async def control_start_get_slash(source_id: str = Query("")) -> JSONResponse:
    return await control_start_get(source_id=source_id)


@app.post("/api/control/stop")
async def control_stop(payload: Dict[str, str]) -> JSONResponse:
    source_id = (payload.get("source_id") or "").strip()
    run_id = (payload.get("run_id") or "").strip()
    if not source_id:
        raise HTTPException(status_code=400, detail="source_id 不能为空")

    stopped = _stop_run(source_id, run_id=run_id)
    _clear_detect_cache(source_id)
    _clear_infer_cache(source_id)
    logger.info(
        "控制停止 source=%s requested_run=%s stopped=%s",
        _short_id(source_id),
        _short_id(run_id),
        stopped,
    )
    return JSONResponse(
        {
            "source_id": source_id,
            "run_id": run_id,
            "stopped": stopped,
            "message": "已请求停止",
        }
    )


@app.get("/api/control/stop")
async def control_stop_get(
    source_id: str = Query(""),
    run_id: str = Query(""),
) -> JSONResponse:
    return await control_stop({"source_id": source_id, "run_id": run_id})


@app.post("/api/control/stop-all")
async def control_stop_all() -> JSONResponse:
    stopped_sources = _stop_all_runs(reason="stop_all_api")
    for sid in stopped_sources:
        _clear_detect_cache(sid)
        _clear_infer_cache(sid)
    logger.info("全量停止 API 调用 stopped=%d", len(stopped_sources))
    return JSONResponse(
        {
            "stopped_sources": stopped_sources,
            "count": len(stopped_sources),
            "message": "已停止所有分析会话",
        }
    )


@app.get("/api/stream/{source_id}")
async def video_stream(
    request: Request,
    source_id: str,
    run_id: str = Query(...),
    mode: str = Query("infer", pattern="^(infer|detect)$"),
    targets: str = Query(""),
):
    if source_id not in SOURCES:
        raise HTTPException(status_code=404, detail="source_id 不存在")
    active_run = _get_active_run_id(source_id)
    if not active_run:
        raise HTTPException(status_code=409, detail="分析会话未激活或已停止")
    if run_id != active_run:
        logger.info(
            "视频流忽略过期 run source=%s requested=%s active=%s",
            _short_id(source_id),
            _short_id(run_id),
            _short_id(active_run),
        )
        return Response(status_code=204)

    target_list = [x.strip() for x in targets.split(",") if x.strip()]
    logger.info(
        "视频流请求 source=%s run=%s mode=%s targets=%s detail=%s",
        _short_id(source_id),
        _short_id(run_id),
        mode,
        ",".join(target_list) if target_list else "-",
        _source_brief(source_id),
    )

    async def gen_frames():
        loop = asyncio.get_event_loop()
        reader: Optional[ThreadedFrameReader] = None
        open_fail = False

        try:
            cap = await loop.run_in_executor(None, _open_capture, source_id)
        except Exception as exc:
            logger.warning("打开视频源失败: %s", exc)
            open_fail = True
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1:
            fps = float(STREAM_TARGET_FPS)

        is_file = SOURCES.get(source_id, {}).get("kind") == "file"
        base_target_fps = float(
            DETECT_STREAM_TARGET_FPS if mode == "detect" else STREAM_TARGET_FPS
        )
        if mode == "detect" and (not is_file):
            # Live-stream detection is CPU/GPU heavy; cap render FPS harder for smoothness.
            base_target_fps = min(base_target_fps, float(LIVE_DETECT_STREAM_TARGET_FPS))
        elif mode == "infer" and (not is_file):
            base_target_fps = min(base_target_fps, float(LIVE_STREAM_TARGET_FPS))

        target_fps = min(base_target_fps, fps)
        frame_step = max(0, int(round(fps / target_fps)) - 1) if fps > target_fps else 0
        if mode == "detect" and (not is_file):
            frame_step = max(frame_step, LIVE_DETECT_MIN_FRAME_STEP)
        delay = 1.0 / target_fps
        detect_interval = (
            YOLO_LIVE_STREAM_INFER_INTERVAL_SEC
            if (mode == "detect" and (not is_file))
            else YOLO_STREAM_INFER_INTERVAL_SEC
        )
        detect_imgsz = YOLO_LIVE_STREAM_IMGSZ if (mode == "detect" and (not is_file)) else YOLO_STREAM_IMGSZ
        display_max_edge = (
            LIVE_DETECT_STREAM_MAX_EDGE
            if (mode == "detect" and (not is_file))
            else (DETECT_STREAM_MAX_EDGE if mode == "detect"
                  else (LIVE_STREAM_MAX_EDGE if (not is_file) else STREAM_MAX_EDGE))
        )
        display_jpeg_quality = (
            LIVE_DETECT_JPEG_QUALITY
            if (mode == "detect" and (not is_file))
            else (LIVE_STREAM_JPEG_QUALITY if (mode == "infer" and (not is_file))
                  else STREAM_JPEG_QUALITY)
        )

        # Use threaded reader for smoother frame acquisition
        if mode == "detect":
            # Live streams prefer smaller queues to avoid backlog and bursty playback.
            q_size = 3 if not is_file else 8
            # File playback: use backpressure to avoid frame drops (smoother).
            # Live streams: drop old frames to keep latency low.
            drop = not is_file
        else:
            q_size = 6 if is_file else 10
            drop = not is_file
        reader = ThreadedFrameReader(
            cap,
            queue_size=q_size,
            skip_frames=frame_step,
            is_file=is_file,
            drop_if_full=drop,
            source_id=source_id,
        )
        logger.info(
            "视频流启动 source=%s run=%s mode=%s src_fps=%.2f target_fps=%.2f step=%d qsize=%d is_file=%s detect_interval=%.2fs detect_imgsz=%d edge=%d q=%d",
            _short_id(source_id),
            _short_id(run_id),
            mode,
            fps,
            target_fps,
            frame_step,
            reader.queue.maxsize,
            is_file,
            detect_interval,
            detect_imgsz,
            display_max_edge,
            display_jpeg_quality,
        )

        frame_deadline = time.monotonic()
        sync_start_wall = None
        sync_start_pos_msec = None
        frame_idx = 0
        last_dets: List[Dict[str, object]] = []
        detect_future = None
        detect_future_frame_idx = 0
        last_detect_submit_ts = 0.0
        detect_future_submit_ts = 0.0
        detect_done_count = 0
        detect_latency_sum = 0.0
        stream_start = time.monotonic()
        stat_last_log = stream_start
        encoded_count = 0
        miss_reads = 0

        try:
            while True:
                if not _is_run_active(source_id, run_id):
                    break
                if await request.is_disconnected():
                    logger.info(
                        "视频流客户端断开 source=%s run=%s",
                        _short_id(source_id),
                        _short_id(run_id),
                    )
                    break

                ret, frame, pos_msec = reader.read()
                if not ret or frame is None:
                    if reader.stopped:
                        break
                    miss_reads += 1
                    # reader.read() already blocks up to 15ms; yield control briefly.
                    await asyncio.sleep(0)
                    continue

                frame_idx += 1
                frame_pos_msec = pos_msec
                if frame_pos_msec < 0 and is_file and fps > 1:
                    frame_pos_msec = (frame_idx - 1) * (1000.0 / fps)
                if mode == "infer" and frame_idx % INFER_CACHE_EVERY_N_FRAMES == 0:
                    infer_frame = _resize_by_max_edge(frame, max_edge=QWEN_MAX_IMAGE_EDGE)
                    with _infer_frame_cache_lock:
                        _infer_frame_cache[source_id] = {
                            "run_id": run_id,
                            "frame": frame_idx,
                            "image": infer_frame.copy(),
                            "ts": time.time(),
                        }
                if mode == "detect":
                    if detect_future is not None and detect_future.done():
                        try:
                            last_dets = detect_future.result()
                            with _detect_cache_lock:
                                _detect_cache[source_id] = {
                                    "run_id": run_id,
                                    "frame": detect_future_frame_idx,
                                    "dets": last_dets,
                                    "ts": time.time(),
                                }
                            if detect_future_submit_ts > 0:
                                detect_done_count += 1
                                detect_latency_sum += max(
                                    0.0, time.monotonic() - detect_future_submit_ts
                                )
                        except Exception as exc:
                            logger.exception("YOLO 检测失败: %s", exc)
                            last_dets = []
                        finally:
                            detect_future = None

                    now = time.monotonic()
                    if (
                        detect_future is None
                        and frame_idx % YOLO_DRAW_EVERY_N_FRAMES == 0
                        and (now - last_detect_submit_ts) >= detect_interval
                    ):
                        detect_future_frame_idx = frame_idx
                        # Use resized frame for YOLO detection
                        detect_input = _resize_by_max_edge(frame, max_edge=detect_imgsz)
                        detect_future = loop.run_in_executor(
                            _yolo_executor,
                            _yolo_detect,
                            detect_input,
                            target_list,
                            detect_imgsz,
                        )
                        last_detect_submit_ts = now
                        detect_future_submit_ts = now

                display_frame, encoded_bytes = await loop.run_in_executor(
                    _encode_executor,
                    _resize_draw_encode_jpeg,
                    frame,
                    display_max_edge,
                    display_jpeg_quality,
                    last_dets if mode == "detect" else None,
                )
                if not encoded_bytes:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded_bytes
                    + b"\r\n"
                )
                encoded_count += 1

                now_log = time.monotonic()
                if now_log - stat_last_log >= 5.0:
                    elapsed = max(1e-6, now_log - stream_start)
                    out_fps = encoded_count / elapsed
                    logger.info(
                        "视频流统计 source=%s run=%s mode=%s frames=%d out_fps=%.2f miss_reads=%d queue=%d detect_done=%d detect_avg=%.3fs",
                        _short_id(source_id),
                        _short_id(run_id),
                        mode,
                        encoded_count,
                        out_fps,
                        miss_reads,
                        reader.queue.qsize() if reader else -1,
                        detect_done_count,
                        (detect_latency_sum / detect_done_count)
                        if detect_done_count > 0
                        else 0.0,
                    )
                    stat_last_log = now_log

                # File sources use timestamp pacing; live streams use deadline pacing
                # because network timestamps are often unstable.
                if is_file and frame_pos_msec >= 0:
                    if sync_start_wall is None or sync_start_pos_msec is None:
                        sync_start_wall = time.monotonic()
                        sync_start_pos_msec = frame_pos_msec
                    expected_elapsed = max(
                        0.0, (frame_pos_msec - sync_start_pos_msec) / 1000.0
                    )
                    remaining = (sync_start_wall + expected_elapsed) - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    elif remaining < -0.8:
                        # Too far behind; realign to avoid long-term drift.
                        sync_start_wall = time.monotonic() - expected_elapsed
                else:
                    # Fallback pacing by target FPS when timestamps are unavailable.
                    frame_deadline += delay
                    remaining = frame_deadline - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    elif remaining < -0.15:
                        frame_deadline = time.monotonic()
        finally:
            if detect_future is not None and (not detect_future.done()):
                detect_future.cancel()
            if reader:
                # Run cleanup in a background thread to avoid blocking the
                # asyncio event loop (reader.stop joins the read thread which
                # can block for several seconds on network sources).
                _reader_ref = reader
                threading.Thread(
                    target=_reader_ref.stop, daemon=True, name="reader-cleanup"
                ).start()
            if not open_fail:
                total = max(1e-6, time.monotonic() - stream_start)
                logger.info(
                    "视频流结束 source=%s run=%s mode=%s duration=%.2fs frames=%d avg_fps=%.2f miss_reads=%d detect_done=%d detect_avg=%.3fs",
                    _short_id(source_id),
                    _short_id(run_id),
                    mode,
                    total,
                    encoded_count,
                    encoded_count / total,
                    miss_reads,
                    detect_done_count,
                    (detect_latency_sum / detect_done_count)
                    if detect_done_count > 0
                    else 0.0,
                )

    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/infer/stream")
async def infer_stream(
    request: Request,
    source_id: str,
    run_id: str,
    prompt: str = DEFAULT_PROMPT,
):
    if source_id not in SOURCES:
        raise HTTPException(status_code=404, detail="source_id 不存在")
    active_run = _get_active_run_id(source_id)
    if not active_run:
        raise HTTPException(status_code=409, detail="分析会话未激活或已停止")
    if run_id != active_run:
        logger.info(
            "推理流忽略过期 run source=%s requested=%s active=%s",
            _short_id(source_id),
            _short_id(run_id),
            _short_id(active_run),
        )
        return Response(status_code=204)
    logger.info(
        "推理流请求 source=%s run=%s prompt_len=%d detail=%s",
        _short_id(source_id),
        _short_id(run_id),
        len(prompt or ""),
        _source_brief(source_id),
    )

    async def event_gen():
        loop = asyncio.get_event_loop()
        cap: Optional[cv2.VideoCapture] = None
        use_dedicated_capture = False
        source_kind = SOURCES.get(source_id, {}).get("kind", "")
        frame_source = "shared_cache"
        cache_wait_start = time.monotonic()
        last_cache_frame_idx = -1
        fps = 12.0
        sample_every_n = 1
        frame_idx = 0
        max_loop_retries = 10
        retry_count = 0
        last_infer_ts = 0.0
        infer_count = 0
        infer_latency_sum = 0.0
        stream_start = time.monotonic()
        stat_last_log = stream_start

        try:
            while True:
                if await request.is_disconnected():
                    break
                if not _is_run_active(source_id, run_id):
                    break

                frame = None
                infer_frame_idx = -1
                if not use_dedicated_capture:
                    cached = await loop.run_in_executor(
                        None,
                        _get_latest_infer_cache_frame,
                        source_id,
                        run_id,
                        last_cache_frame_idx,
                    )
                    if cached is not None:
                        infer_frame_idx, frame = cached
                        last_cache_frame_idx = infer_frame_idx
                    else:
                        wait_elapsed = time.monotonic() - cache_wait_start
                        if wait_elapsed >= INFER_CACHE_WAIT_SEC:
                            # For URL streams, avoid opening a second decoder in infer SSE.
                            # Reusing cached frames keeps pressure lower in dual-stream sessions.
                            if source_kind == "url":
                                await asyncio.sleep(INFER_CACHE_POLL_INTERVAL_SEC)
                                continue
                            use_dedicated_capture = True
                            frame_source = "dedicated_capture"
                            try:
                                cap = await loop.run_in_executor(
                                    None, _open_capture, source_id
                                )
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                if not fps or np.isnan(fps) or fps <= 1:
                                    fps = 12.0
                                sample_every_n = max(1, int(fps * 1.4))
                                logger.info(
                                    "推理流回退到独立解码 source=%s run=%s fps=%.2f sample_every_n=%d",
                                    _short_id(source_id),
                                    _short_id(run_id),
                                    fps,
                                    sample_every_n,
                                )
                            except ValueError as exc:
                                yield f"data: {json.dumps({'type': 'error', 'text': str(exc)}, ensure_ascii=False)}\n\n"
                                return
                        await asyncio.sleep(INFER_CACHE_POLL_INTERVAL_SEC)
                        continue
                else:
                    if cap is None:
                        await asyncio.sleep(INFER_CACHE_POLL_INTERVAL_SEC)
                        continue
                    ret, frame = await loop.run_in_executor(None, cap.read)
                    if not ret:
                        if SOURCES.get(source_id, {}).get("kind") == "file":
                            retry_count += 1
                            if retry_count > max_loop_retries:
                                break
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            await asyncio.sleep(0.15)
                            continue
                        await asyncio.sleep(0.2)
                        continue

                    retry_count = 0
                    frame_idx += 1
                    if frame_idx % sample_every_n != 0:
                        await asyncio.sleep(0.01)
                        continue
                    infer_frame_idx = frame_idx
                    frame = _resize_by_max_edge(frame, max_edge=QWEN_MAX_IMAGE_EDGE)

                if frame is None:
                    await asyncio.sleep(INFER_CACHE_POLL_INTERVAL_SEC)
                    continue
                # Dynamically scale inference interval by active run count
                # so panels don't starve each other waiting on _qwen_lock.
                with _active_runs_lock:
                    _n_active = max(1, len(_active_runs))
                _dynamic_interval = INFER_MIN_INTERVAL_SEC * _n_active
                if (time.monotonic() - last_infer_ts) < _dynamic_interval:
                    await asyncio.sleep(0.03)
                    continue
                last_infer_ts = time.monotonic()

                try:
                    infer_t0 = time.monotonic()
                    text = await loop.run_in_executor(
                        None, _qwen_caption, frame, prompt, source_id, run_id
                    )
                    infer_latency = time.monotonic() - infer_t0
                    infer_count += 1
                    infer_latency_sum += infer_latency
                except Exception as exc:
                    logger.exception("VLM-Online 推理失败: %s", exc)
                    yield (
                        f"data: {json.dumps({'type': 'error', 'text': f'VLM-Online 推理失败: {exc}'}, ensure_ascii=False)}\n\n"
                    )
                    await asyncio.sleep(0.2)
                    continue

                if not _is_run_active(source_id, run_id):
                    break
                yield f"data: {json.dumps({'type': 'start', 'text': '', 'frame': infer_frame_idx}, ensure_ascii=False)}\n\n"

                for piece in text.split("，"):
                    if await request.is_disconnected() or (not _is_run_active(source_id, run_id)):
                        return
                    chunk = piece + "，"
                    yield (
                        f"data: {json.dumps({'type': 'chunk', 'text': chunk, 'frame': infer_frame_idx}, ensure_ascii=False)}\n\n"
                    )
                    await asyncio.sleep(0.08)

                yield (
                    f"data: {json.dumps({'type': 'end', 'text': '', 'frame': infer_frame_idx, 'ts': time.time()}, ensure_ascii=False)}\n\n"
                )
                now_log = time.monotonic()
                if now_log - stat_last_log >= 6.0:
                    elapsed = max(1e-6, now_log - stream_start)
                    avg_latency = infer_latency_sum / max(1, infer_count)
                    logger.info(
                        "推理流统计 source=%s run=%s source_mode=%s inferences=%d infer_rate=%.2f/s avg_latency=%.2fs",
                        _short_id(source_id),
                        _short_id(run_id),
                        frame_source,
                        infer_count,
                        infer_count / elapsed,
                        avg_latency,
                    )
                    stat_last_log = now_log
                await asyncio.sleep(0.06)
        finally:
            if cap is not None:
                cap.release()
            elapsed = max(1e-6, time.monotonic() - stream_start)
            avg_latency = infer_latency_sum / max(1, infer_count)
            logger.info(
                "推理流结束 source=%s run=%s source_mode=%s duration=%.2fs inferences=%d infer_rate=%.2f/s avg_latency=%.2fs",
                _short_id(source_id),
                _short_id(run_id),
                frame_source,
                elapsed,
                infer_count,
                infer_count / elapsed,
                avg_latency,
            )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/detect/stream")
async def detect_stream(
    request: Request,
    source_id: str,
    run_id: str,
    targets: str = "",
):
    if source_id not in SOURCES:
        raise HTTPException(status_code=404, detail="source_id 不存在")
    active_run = _get_active_run_id(source_id)
    if not active_run:
        raise HTTPException(status_code=409, detail="分析会话未激活或已停止")
    if run_id != active_run:
        logger.info(
            "检测流忽略过期 run source=%s requested=%s active=%s",
            _short_id(source_id),
            _short_id(run_id),
            _short_id(active_run),
        )
        return Response(status_code=204)

    _ = [x.strip() for x in targets.split(",") if x.strip()]
    logger.info(
        "检测事件流请求 source=%s run=%s targets=%s detail=%s",
        _short_id(source_id),
        _short_id(run_id),
        targets or "-",
        _source_brief(source_id),
    )

    async def event_gen():
        last_sent_frame = -1
        idle_ticks = 0
        sent_messages = 0
        stream_start = time.monotonic()
        stat_last_log = stream_start
        while True:
            if await request.is_disconnected():
                break
            if not _is_run_active(source_id, run_id):
                break

            with _detect_cache_lock:
                state = _detect_cache.get(source_id)

            if (
                state
                and state.get("run_id") == run_id
                and state.get("frame", -1) != last_sent_frame
            ):
                frame_idx = int(state.get("frame", -1))
                dets = state.get("dets", [])
                if dets:
                    summary = ", ".join(
                        [f"{d['label']}({d['conf']:.2f})" for d in dets]
                    )
                    message = f"[帧{frame_idx}] 检测到: {summary}"
                else:
                    message = f"[帧{frame_idx}] 未检测到目标"

                yield (
                    f"data: {json.dumps({'type': 'detect', 'text': message, 'frame': frame_idx, 'count': len(dets)}, ensure_ascii=False)}\n\n"
                )
                sent_messages += 1
                last_sent_frame = frame_idx
                idle_ticks = 0
            else:
                idle_ticks += 1
                if idle_ticks % 80 == 0:
                    yield (
                        f"data: {json.dumps({'type': 'waiting', 'text': '等待检测结果...', 'frame': -1, 'count': 0}, ensure_ascii=False)}\n\n"
                    )

            now_log = time.monotonic()
            if now_log - stat_last_log >= 6.0:
                elapsed = max(1e-6, now_log - stream_start)
                logger.info(
                    "检测事件流统计 source=%s run=%s messages=%d rate=%.2f/s idle_ticks=%d",
                    _short_id(source_id),
                    _short_id(run_id),
                    sent_messages,
                    sent_messages / elapsed,
                    idle_ticks,
                )
                stat_last_log = now_log
            await asyncio.sleep(0.08)
        elapsed = max(1e-6, time.monotonic() - stream_start)
        logger.info(
            "检测事件流结束 source=%s run=%s duration=%.2fs messages=%d rate=%.2f/s",
            _short_id(source_id),
            _short_id(run_id),
            elapsed,
            sent_messages,
            sent_messages / elapsed,
        )

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
