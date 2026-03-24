# Video Inference Web Demo

一个前后端一体化的在线视频推理 demo，支持两种模式：

- `Qwen 实时推理`：对视频流按时间采样帧，调用 `Qwen2.5-VL-3B` 输出流式文本。
- `YOLO 实时检测`：对视频流实时调用 `YOLO` 进行目标检测并绘制 box。

支持两类视频输入：

- 上传离线视频文件
- 输入实时摄像头/流媒体 URL（如 `rtsp/http/hls`）

## 功能点

- 左侧实时视频画面（MJPEG 流）
- 右侧流式输出面板（可滚动，最新内容高亮）
- Prompt 实时编辑 + 一键恢复默认
- 检测目标可自定义输入（例如 `person,car,bicycle`）
- 推理/检测两种模式可切换
- 中文/英文展示友好

## 目录结构

```text
web_vlm/
├── app.py
├── requirements.txt
├── uploads/
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

## 启动方式

1. 安装依赖

```bash
cd web_vlm
conda create -n web_vlm python=3.10 -y
conda activate web_vlm
# 先安装与你服务器 CUDA 匹配的 torch（示例，按你的 CUDA 版本调整）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

2. 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

3. 浏览器打开

```text
http://127.0.0.1:8000
```

## 模型路径（硬编码）

在 `app.py` 顶部直接修改：

- `QWEN_MODEL_PATH`：Qwen2.5-VL-3B 模型目录
- `YOLO_MODEL_PATH`：YOLO 权重文件（`.pt`）
- `GPU_DEVICE`：GPU 设备（默认 `cuda:0`）

当前代码要求使用 GPU；若无 CUDA 会报错。

## 接口说明（核心）

- `POST /api/source/upload`：上传视频
- `POST /api/source/url`：注册在线视频 URL
- `GET /api/stream/{source_id}?mode=infer|detect&targets=...`：视频 MJPEG 流
- `GET /api/infer/stream?source_id=...&prompt=...`：推理 SSE 流式输出
- `GET /api/detect/stream?source_id=...&targets=...`：检测 SSE 事件流
