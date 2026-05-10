#!/usr/bin/env python3
"""
Quick stream availability test for OpenCV/FFmpeg.

Example:
  python test_stream.py "rtsp://xxx" --seconds 20 --transport tcp
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2


BACKEND_CANDIDATES = [
    ("FFMPEG", cv2.CAP_FFMPEG),
    ("DEFAULT", cv2.CAP_ANY),
]


def build_capture(
    source: str,
    transport: str,
    timeout_ms: int,
    backend_id: int,
) -> cv2.VideoCapture:
    if source.startswith("rtsp://") and transport != "auto":
        timeout_us = max(1, timeout_ms) * 1000
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}|stimeout;{timeout_us}"
        )
    return cv2.VideoCapture(source, backend_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test if a stream can be played by OpenCV.")
    parser.add_argument("source", help="Stream URL or local video path")
    parser.add_argument("--seconds", type=int, default=15, help="Test duration in seconds")
    parser.add_argument(
        "--transport",
        choices=["auto", "tcp", "udp"],
        default="tcp",
        help="RTSP transport for FFmpeg",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=10000,
        help="Open timeout in ms for FFmpeg options",
    )
    parser.add_argument(
        "--save-frame",
        default="",
        help="Optional path to save first successfully decoded frame",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window (for local desktop debugging)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ffmpeg", "default"],
        default="auto",
        help="Capture backend strategy",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.strip()
    if not source:
        print("[FAIL] empty source", file=sys.stderr)
        return 2

    print("=== Stream Test Start ===")
    print(f"source      : {source}")
    print(f"duration    : {args.seconds}s")
    print(f"transport   : {args.transport}")
    print(f"timeout-ms  : {args.timeout_ms}")

    backend_plan = []
    if args.backend == "ffmpeg":
        backend_plan = [("FFMPEG", cv2.CAP_FFMPEG)]
    elif args.backend == "default":
        backend_plan = [("DEFAULT", cv2.CAP_ANY)]
    else:
        backend_plan = BACKEND_CANDIDATES

    cap = None
    open_cost = 0.0
    used_backend = ""
    for backend_name, backend_id in backend_plan:
        t0 = time.time()
        cap_try = build_capture(source, args.transport, args.timeout_ms, backend_id)
        cost = time.time() - t0
        print(f"[INFO] try backend={backend_name}, open_cost={cost:.2f}s")
        if cap_try.isOpened():
            cap = cap_try
            open_cost = cost
            used_backend = backend_name
            break
        cap_try.release()

    if cap is None:
        print("[FAIL] cannot open stream with available backends")
        return 2

    expected_fps = cap.get(cv2.CAP_PROP_FPS)
    if not expected_fps or expected_fps <= 0:
        expected_fps = 0.0

    print(
        f"[OK] opened (backend={used_backend}, open_cost={open_cost:.2f}s, reported_fps={expected_fps:.2f})"
    )

    success = 0
    failures = 0
    consec_fail = 0
    first_ok_ts = 0.0
    saved_frame = False
    width = 0
    height = 0

    deadline = time.time() + max(1, args.seconds)
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            success += 1
            consec_fail = 0
            if first_ok_ts == 0.0:
                first_ok_ts = time.time()
            height, width = frame.shape[:2]

            if args.save_frame and not saved_frame:
                save_path = Path(args.save_frame)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), frame)
                saved_frame = True
                print(f"[INFO] saved sample frame: {save_path}")

            if args.show:
                cv2.imshow("stream_test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            failures += 1
            consec_fail += 1
            if consec_fail > 120:
                print("[WARN] too many consecutive decode failures, stopping early")
                break
            time.sleep(0.01)

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = max(1e-6, time.time() - t0)
    active_elapsed = max(1e-6, time.time() - first_ok_ts) if first_ok_ts > 0 else elapsed
    read_fps = success / active_elapsed
    total = success + failures
    success_ratio = success / total if total > 0 else 0.0

    print("=== Stream Test Summary ===")
    print(f"decoded_frames : {success}")
    print(f"failed_reads   : {failures}")
    print(f"success_ratio  : {success_ratio:.2%}")
    print(f"read_fps       : {read_fps:.2f}")
    print(f"resolution     : {width}x{height}" if width and height else "resolution     : unknown")

    passed = success >= 20 and success_ratio >= 0.5
    if passed:
        print("[PASS] stream is playable for current OpenCV/FFmpeg environment")
        return 0

    print("[FAIL] stream decode is unstable or unavailable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
