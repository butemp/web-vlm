import asyncio
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

logger = logging.getLogger("web_vlm")

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
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB

# ---------------------------------------------------------------------------
# Hardcoded model paths (edit here before deploying).
# ---------------------------------------------------------------------------
QWEN_MODEL_PATH = "/data/models/Qwen2.5-VL-3B-Instruct"
YOLO_MODEL_PATH = "/data/models/yolov8x.pt"

GPU_DEVICE = "cuda:0"
QWEN_MAX_NEW_TOKENS = 96
QWEN_MAX_IMAGE_EDGE = 640
YOLO_CONF = 0.25
YOLO_IMGSZ = 640
YOLO_STREAM_IMGSZ = 416
YOLO_DRAW_EVERY_N_FRAMES = 3
YOLO_STREAM_INFER_INTERVAL_SEC = 0.45
STREAM_MAX_EDGE = 720
STREAM_JPEG_QUALITY = 65
STREAM_TARGET_FPS = 25
INFER_CACHE_EVERY_N_FRAMES = 2
INFER_MIN_INTERVAL_SEC = 1.2
INFER_CACHE_MAX_AGE_SEC = 2.0
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

app = FastAPI(title="Realtime Video Inference Demo", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def _startup_preload_models() -> None:
    if not PRELOAD_QWEN_ON_STARTUP:
        return

    logger.info("服务启动：开始预加载 Qwen 模型")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _ensure_qwen_loaded)
        logger.info("服务启动：Qwen 模型预加载完成")
    except Exception as exc:
        logger.exception("服务启动：Qwen 模型预加载失败: %s", exc)
        raise RuntimeError(f"Qwen 模型预加载失败: {exc}") from exc

# ---------------------------------------------------------------------------
# In-memory registry for uploaded files / online stream URLs.
# ---------------------------------------------------------------------------
SOURCES: Dict[str, Dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _open_capture(source_id: str) -> cv2.VideoCapture:
    source = SOURCES.get(source_id)
    if not source:
        raise ValueError("source_id not found")
    value = source["value"]

    ffmpeg_options = _build_ffmpeg_capture_options(value)
    if ffmpeg_options:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_options

    cap = cv2.VideoCapture(value, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(value, cv2.CAP_ANY)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频源: {value}")
    # Optimize capture settings for smoother playback
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    # For network streams, set longer timeouts
    if value.startswith(("http://", "https://", "rtsp://")):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    return cap


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    if not name:
        name = "video.mp4"
    return name


def _start_run(source_id: str) -> str:
    run_id = str(uuid.uuid4())
    with _active_runs_lock:
        _active_runs[source_id] = run_id
    return run_id


def _is_run_active(source_id: str, run_id: str) -> bool:
    with _active_runs_lock:
        return _active_runs.get(source_id) == run_id


def _stop_run(source_id: str, run_id: str = "") -> bool:
    with _active_runs_lock:
        current = _active_runs.get(source_id)
        if current is None:
            return False
        if run_id and current != run_id:
            return False
        _active_runs.pop(source_id, None)
        return True


def _clear_detect_cache(source_id: str) -> None:
    with _detect_cache_lock:
        _detect_cache.pop(source_id, None)


def _clear_infer_cache(source_id: str) -> None:
    with _infer_frame_cache_lock:
        _infer_frame_cache.pop(source_id, None)


def _build_ffmpeg_capture_options(url: str) -> str:
    lower = url.lower()
    if lower.startswith("rtsp://"):
        return "rtsp_transport;tcp|stimeout;10000000|fflags;nobuffer|flags;low_delay"
    if ".m3u8" in lower:
        # HLS stream: ultra low latency settings for live streaming
        return (
            "fflags;nobuffer+discardcorrupt|flags;low_delay|"
            "analyzeduration;500000|probesize;500000|"
            "reconnect;1|reconnect_streamed;1|reconnect_delay_max;2|"
            "rw_timeout;15000000|timeout;10000000"
        )
    return ""


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
            raise RuntimeError(f"Qwen 模型路径不存在: {QWEN_MODEL_PATH}")

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("开始加载 Qwen 模型: %s", QWEN_MODEL_PATH)
        _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto",
        )
        _qwen_model.eval()
        _qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
        logger.info("Qwen 模型加载完成")

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
            raise RuntimeError(f"YOLO 模型路径不存在: {YOLO_MODEL_PATH}")

        logger.info("开始加载 YOLO 模型: %s", YOLO_MODEL_PATH)
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLO 模型加载完成")

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

    def __init__(
        self,
        cap: cv2.VideoCapture,
        queue_size: int = 8,
        skip_frames: int = 0,
        is_file: bool = False,
        drop_if_full: bool = True,
    ):
        self.cap = cap
        self.skip_frames = skip_frames
        self.is_file = is_file
        self.drop_if_full = drop_if_full
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        retry_count = 0
        max_retries = 3
        while not self.stopped:
            if self.queue.full():
                if self.drop_if_full:
                    # Live streams prefer low latency: drop oldest frame when backlogged.
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
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
                        self.queue.put((False, None, -1.0))
                        return
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Network stream failed, try to continue
                    time.sleep(0.05)
                    continue

            retry_count = 0
            pos_msec = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            if np.isnan(pos_msec) or pos_msec < 0:
                pos_msec = -1.0
            try:
                self.queue.put((True, frame, pos_msec), timeout=0.1)
            except queue.Full:
                pass

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        if self.stopped and self.queue.empty():
            return False, None, -1.0
        try:
            return self.queue.get(timeout=0.5)
        except queue.Empty:
            return False, None, -1.0

    def stop(self):
        self.stopped = True
        self.cap.release()


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


def _qwen_caption(frame: np.ndarray, prompt: str, source_id: str, run_id: str) -> str:
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

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList(
                    [_RunStopCriteria(source_id, run_id)]
                ),
            )

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

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["xyxy"]
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
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/defaults")
async def defaults() -> JSONResponse:
    return JSONResponse({"default_prompt": DEFAULT_PROMPT})


# ---------------------------------------------------------------------------
# Chunked Upload — reliable through tunnels / slow connections
# ---------------------------------------------------------------------------
_upload_sessions: Dict[str, Dict[str, object]] = {}
_upload_sessions_lock = threading.Lock()


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

    return JSONResponse({"upload_id": upload_id, "message": "上传会话已创建"})


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

    return JSONResponse({
        "chunk_index": chunk_index,
        "received": received_count,
        "total_chunks": session["total_chunks"],
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

    run_id = _start_run(source_id)
    _clear_detect_cache(source_id)
    _clear_infer_cache(source_id)
    return JSONResponse(
        {
            "source_id": source_id,
            "run_id": run_id,
            "message": "分析会话已启动",
        }
    )


@app.get("/api/control/start")
async def control_start_get(source_id: str = Query("")) -> JSONResponse:
    return await control_start({"source_id": source_id})


@app.post("/api/control/stop")
async def control_stop(payload: Dict[str, str]) -> JSONResponse:
    source_id = (payload.get("source_id") or "").strip()
    run_id = (payload.get("run_id") or "").strip()
    if not source_id:
        raise HTTPException(status_code=400, detail="source_id 不能为空")

    stopped = _stop_run(source_id, run_id=run_id)
    _clear_detect_cache(source_id)
    _clear_infer_cache(source_id)
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


@app.get("/api/stream/{source_id}")
async def video_stream(
    source_id: str,
    run_id: str = Query(...),
    mode: str = Query("infer", pattern="^(infer|detect)$"),
    targets: str = Query(""),
):
    if source_id not in SOURCES:
        raise HTTPException(status_code=404, detail="source_id 不存在")
    if not _is_run_active(source_id, run_id):
        raise HTTPException(status_code=409, detail="分析会话未激活或已停止")

    target_list = [x.strip() for x in targets.split(",") if x.strip()]

    async def gen_frames():
        loop = asyncio.get_event_loop()
        reader: Optional[ThreadedFrameReader] = None

        try:
            cap = await loop.run_in_executor(None, _open_capture, source_id)
        except ValueError as exc:
            logger.warning("打开视频源失败: %s", exc)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1:
            fps = float(STREAM_TARGET_FPS)

        target_fps = min(float(STREAM_TARGET_FPS), fps)
        frame_step = max(0, int(round(fps / target_fps)) - 1) if fps > target_fps else 0
        delay = 1.0 / target_fps
        is_file = SOURCES.get(source_id, {}).get("kind") == "file"
        if is_file and mode == "detect":
            # In detection mode for offline files, avoid extra skip to prevent fast-forward feel.
            frame_step = 0

        # Use threaded reader for smoother frame acquisition
        reader = ThreadedFrameReader(
            cap,
            queue_size=6 if is_file else 10,
            skip_frames=frame_step,
            is_file=is_file,
            drop_if_full=not is_file,
        )

        frame_deadline = time.monotonic()
        sync_start_wall = None
        sync_start_pos_msec = None
        frame_idx = 0
        last_dets: List[Dict[str, object]] = []
        detect_future = None
        detect_future_frame_idx = 0
        last_detect_submit_ts = 0.0

        try:
            while True:
                if not _is_run_active(source_id, run_id):
                    break

                ret, frame, pos_msec = reader.read()
                if not ret or frame is None:
                    if reader.stopped:
                        break
                    continue

                frame_idx += 1
                frame_pos_msec = pos_msec
                if frame_pos_msec < 0 and is_file and fps > 1:
                    frame_pos_msec = (frame_idx - 1) * (1000.0 / fps)
                if mode == "infer" and frame_idx % INFER_CACHE_EVERY_N_FRAMES == 0:
                    # Cache the latest frame for infer SSE without blocking stream timing.
                    cache_frame = await loop.run_in_executor(
                        None, _resize_by_max_edge, frame.copy(), QWEN_MAX_IMAGE_EDGE
                    )
                    with _infer_frame_cache_lock:
                        _infer_frame_cache[source_id] = {
                            "run_id": run_id,
                            "frame": frame_idx,
                            "image": cache_frame,
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
                        except Exception as exc:
                            logger.exception("YOLO 检测失败: %s", exc)
                            last_dets = []
                        finally:
                            detect_future = None

                    now = time.monotonic()
                    if (
                        detect_future is None
                        and frame_idx % YOLO_DRAW_EVERY_N_FRAMES == 0
                        and (now - last_detect_submit_ts) >= YOLO_STREAM_INFER_INTERVAL_SEC
                    ):
                        detect_future_frame_idx = frame_idx
                        # Use resized frame for YOLO detection
                        detect_input = _resize_by_max_edge(frame, max_edge=YOLO_STREAM_IMGSZ)
                        detect_future = loop.run_in_executor(
                            None,
                            _yolo_detect,
                            detect_input,
                            target_list,
                            YOLO_STREAM_IMGSZ,
                        )
                        last_detect_submit_ts = now

                display_frame, encoded_bytes = await loop.run_in_executor(
                    None,
                    _resize_draw_encode_jpeg,
                    frame,
                    STREAM_MAX_EDGE,
                    STREAM_JPEG_QUALITY,
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

                # Prefer timestamp pacing to avoid detect-mode speed-up when source FPS is misreported.
                if frame_pos_msec >= 0:
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
                    elif remaining < -0.5:
                        frame_deadline = time.monotonic()
        finally:
            if reader:
                reader.stop()

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
    if not _is_run_active(source_id, run_id):
        raise HTTPException(status_code=409, detail="分析会话未激活或已停止")

    async def event_gen():
        loop = asyncio.get_event_loop()
        try:
            cap = await loop.run_in_executor(None, _open_capture, source_id)
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)}, ensure_ascii=False)}\n\n"
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1:
            fps = 12.0
        sample_every_n = max(1, int(fps * 1.4))
        frame_idx = 0
        max_loop_retries = 3
        retry_count = 0
        last_infer_ts = 0.0

        try:
            while True:
                if await request.is_disconnected():
                    break
                if not _is_run_active(source_id, run_id):
                    break

                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    if SOURCES.get(source_id, {}).get("kind") == "file":
                        retry_count += 1
                        if retry_count > max_loop_retries:
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    await asyncio.sleep(0.2)
                    continue

                retry_count = 0
                frame_idx += 1
                if frame_idx % sample_every_n != 0:
                    await asyncio.sleep(0.01)
                    continue
                if (time.monotonic() - last_infer_ts) < INFER_MIN_INTERVAL_SEC:
                    await asyncio.sleep(0.03)
                    continue
                last_infer_ts = time.monotonic()

                frame = _resize_by_max_edge(frame, max_edge=QWEN_MAX_IMAGE_EDGE)
                try:
                    text = await loop.run_in_executor(
                        None, _qwen_caption, frame.copy(), prompt, source_id, run_id
                    )
                except Exception as exc:
                    logger.exception("Qwen 推理失败: %s", exc)
                    yield (
                        f"data: {json.dumps({'type': 'error', 'text': f'Qwen 推理失败: {exc}'}, ensure_ascii=False)}\n\n"
                    )
                    await asyncio.sleep(0.2)
                    continue

                if not _is_run_active(source_id, run_id):
                    break
                yield f"data: {json.dumps({'type': 'start', 'text': '', 'frame': frame_idx}, ensure_ascii=False)}\n\n"

                for piece in text.split("，"):
                    if await request.is_disconnected() or (not _is_run_active(source_id, run_id)):
                        return
                    chunk = piece + "，"
                    yield (
                        f"data: {json.dumps({'type': 'chunk', 'text': chunk, 'frame': frame_idx}, ensure_ascii=False)}\n\n"
                    )
                    await asyncio.sleep(0.12)

                yield (
                    f"data: {json.dumps({'type': 'end', 'text': '', 'frame': frame_idx, 'ts': time.time()}, ensure_ascii=False)}\n\n"
                )
                await asyncio.sleep(0.06)
        finally:
            cap.release()

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
    if not _is_run_active(source_id, run_id):
        raise HTTPException(status_code=409, detail="分析会话未激活或已停止")

    _ = [x.strip() for x in targets.split(",") if x.strip()]

    async def event_gen():
        last_sent_frame = -1
        idle_ticks = 0
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
                last_sent_frame = frame_idx
                idle_ticks = 0
            else:
                idle_ticks += 1
                if idle_ticks % 80 == 0:
                    yield (
                        f"data: {json.dumps({'type': 'waiting', 'text': '等待检测结果...', 'frame': -1, 'count': 0}, ensure_ascii=False)}\n\n"
                    )

            await asyncio.sleep(0.08)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
