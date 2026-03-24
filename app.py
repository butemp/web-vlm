import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("web_vlm")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULT_PROMPT = "请简单描述一下这个视频"
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB

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


def _frame_signature(frame: np.ndarray) -> Tuple[float, float, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    saturation = float(np.mean(hsv[:, :, 1]))
    hue = float(np.mean(hsv[:, :, 0]))
    return brightness, saturation, hue


def _mock_caption(frame: np.ndarray, prompt: str, frame_idx: int) -> str:
    brightness, saturation, hue = _frame_signature(frame)

    if brightness > 170:
        light_desc = "画面较明亮"
    elif brightness > 110:
        light_desc = "亮度中等"
    else:
        light_desc = "画面偏暗"

    if saturation > 120:
        color_desc = "色彩饱和度较高"
    elif saturation > 70:
        color_desc = "色彩自然"
    else:
        color_desc = "色彩较淡"

    if hue < 50:
        tone_desc = "整体偏暖色调"
    elif hue < 100:
        tone_desc = "整体偏中性色调"
    else:
        tone_desc = "整体偏冷色调"

    return (
        f"[模拟Qwen2.5-VL-3B][帧{frame_idx}] 针对提示词\u201c{prompt}\u201d："
        f"当前{light_desc}，{color_desc}，{tone_desc}，"
        f"场景看起来在持续变化。"
    )


def _mock_detect(
    frame: np.ndarray,
    frame_idx: int,
    targets: List[str],
) -> List[Dict[str, object]]:
    h, w = frame.shape[:2]
    labels = [t.strip() for t in targets if t.strip()] or ["person", "car", "bicycle"]

    detections: List[Dict[str, object]] = []
    for i, label in enumerate(labels[:5]):
        box_w = max(80, int(w * (0.12 + 0.02 * (i % 3))))
        box_h = max(60, int(h * (0.18 + 0.03 * (i % 2))))
        x = int((frame_idx * (11 + i * 3) + i * 137) % max(1, (w - box_w)))
        y = int((frame_idx * (7 + i * 2) + i * 91) % max(1, (h - box_h)))
        conf = round(0.55 + ((frame_idx + i * 13) % 35) / 100, 2)
        detections.append({"label": label, "conf": conf, "xyxy": [x, y, x + box_w, y + box_h]})

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
                    dets = _mock_detect(frame, frame_idx, target_list)
                    frame = _draw_detections(frame, dets)

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

                text = _mock_caption(frame, prompt, frame_idx)
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

                dets = _mock_detect(frame, frame_idx, target_list)
                summary = ", ".join(
                    [f"{d['label']}({d['conf']:.2f})" for d in dets]
                )
                message = f"[帧{frame_idx}] 检测到: {summary}"
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
