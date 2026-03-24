import asyncio
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List

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
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from ultralytics import YOLO
except Exception:
    torch = None
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
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
QWEN_MAX_IMAGE_EDGE = 1024
YOLO_CONF = 0.25
YOLO_IMGSZ = 960
YOLO_DRAW_EVERY_N_FRAMES = 2

# Runtime model cache
_qwen_lock = threading.Lock()
_yolo_lock = threading.Lock()
_qwen_model = None
_qwen_processor = None
_yolo_model = None

app = FastAPI(title="Realtime Video Inference Demo", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
    cap = cv2.VideoCapture(value)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频源: {value}")
    return cap


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    if not name:
        name = "video.mp4"
    return name


def _require_runtime_deps() -> None:
    if torch is None or AutoProcessor is None or YOLO is None:
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


def _resize_for_vlm(frame: np.ndarray, max_edge: int = QWEN_MAX_IMAGE_EDGE) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return frame

    scale = max_edge / float(longest)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _qwen_caption(frame: np.ndarray, prompt: str) -> str:
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


def _yolo_detect(frame: np.ndarray, targets: List[str]) -> List[Dict[str, object]]:
    model = _ensure_yolo_loaded()
    target_set = {x.lower().strip() for x in targets if x.strip()}

    with _yolo_lock:
        results = model.predict(
            source=frame,
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
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


@app.post("/api/source/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    filename = _safe_filename(file.filename or "video.mp4")
    source_id = str(uuid.uuid4())
    target_path = UPLOAD_DIR / f"{source_id}_{filename}"

    total = 0
    with target_path.open("wb") as fp:
        while chunk := await file.read(1024 * 1024):
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
            "message": "视频流地址已注册",
        }
    )


@app.get("/api/stream/{source_id}")
async def video_stream(
    source_id: str,
    mode: str = Query("infer", pattern="^(infer|detect)$"),
    targets: str = Query(""),
):
    if source_id not in SOURCES:
        raise HTTPException(status_code=404, detail="source_id 不存在")

    target_list = [x.strip() for x in targets.split(",") if x.strip()]

    async def gen_frames():
        loop = asyncio.get_event_loop()
        try:
            cap = await loop.run_in_executor(None, _open_capture, source_id)
        except ValueError as exc:
            logger.warning("打开视频源失败: %s", exc)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1:
            fps = 15.0
        delay = 1.0 / min(20.0, max(8.0, fps))
        frame_idx = 0
        last_dets: List[Dict[str, object]] = []
        max_loop_retries = 3
        retry_count = 0

        try:
            while True:
                ret, frame = await loop.run_in_executor(None, cap.read)
                if not ret:
                    if SOURCES.get(source_id, {}).get("kind") == "file":
                        retry_count += 1
                        if retry_count > max_loop_retries:
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                retry_count = 0
                frame_idx += 1
                if mode == "detect":
                    if frame_idx % YOLO_DRAW_EVERY_N_FRAMES == 0:
                        try:
                            dets = await loop.run_in_executor(
                                None, _yolo_detect, frame.copy(), target_list
                            )
                            last_dets = dets
                        except Exception as exc:
                            logger.exception("YOLO 检测失败: %s", exc)
                            last_dets = []
                    frame = _draw_detections(frame, last_dets)

                ok, encoded = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )
                if not ok:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )
                await asyncio.sleep(delay)
        finally:
            cap.release()

    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/infer/stream")
async def infer_stream(
    request: Request,
    source_id: str,
    prompt: str = DEFAULT_PROMPT,
):
    if source_id not in SOURCES:
        raise HTTPException(status_code=404, detail="source_id 不存在")

    async def event_gen():
        loop = asyncio.get_event_loop()
        try:
            cap = await loop.run_in_executor(None, _open_capture, source_id)
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)}, ensure_ascii=False)}\n\n"
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1:
            fps = 15.0

        sample_every_n = max(1, int(fps * 1.5))
        frame_idx = 0
        max_loop_retries = 3
        retry_count = 0

        try:
            while True:
                if await request.is_disconnected():
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

                try:
                    text = await loop.run_in_executor(
                        None, _qwen_caption, frame.copy(), prompt
                    )
                except Exception as exc:
                    logger.exception("Qwen 推理失败: %s", exc)
                    yield (
                        f"data: {json.dumps({'type': 'error', 'text': f'Qwen 推理失败: {exc}'}, ensure_ascii=False)}\n\n"
                    )
                    await asyncio.sleep(0.4)
                    continue
                yield f"data: {json.dumps({'type': 'start', 'text': '', 'frame': frame_idx}, ensure_ascii=False)}\n\n"

                for piece in text.split("，"):
                    if await request.is_disconnected():
                        return
                    chunk = piece + "，"
                    yield (
                        f"data: {json.dumps({'type': 'chunk', 'text': chunk, 'frame': frame_idx}, ensure_ascii=False)}\n\n"
                    )
                    await asyncio.sleep(0.18)

                yield (
                    f"data: {json.dumps({'type': 'end', 'text': '', 'frame': frame_idx, 'ts': time.time()}, ensure_ascii=False)}\n\n"
                )
                await asyncio.sleep(0.2)
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
    targets: str = "",
):
    if source_id not in SOURCES:
        raise HTTPException(status_code=404, detail="source_id 不存在")

    target_list = [x.strip() for x in targets.split(",") if x.strip()]

    async def event_gen():
        loop = asyncio.get_event_loop()
        try:
            cap = await loop.run_in_executor(None, _open_capture, source_id)
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)}, ensure_ascii=False)}\n\n"
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1:
            fps = 15.0
        sample_every_n = max(1, int(fps * 1.0))
        frame_idx = 0
        max_loop_retries = 3
        retry_count = 0

        try:
            while True:
                if await request.is_disconnected():
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

                try:
                    dets = await loop.run_in_executor(
                        None, _yolo_detect, frame.copy(), target_list
                    )
                except Exception as exc:
                    logger.exception("YOLO 检测失败: %s", exc)
                    yield (
                        f"data: {json.dumps({'type': 'error', 'text': f'YOLO 检测失败: {exc}'}, ensure_ascii=False)}\n\n"
                    )
                    await asyncio.sleep(0.3)
                    continue
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
                await asyncio.sleep(0.12)
        finally:
            cap.release()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
