"""HTTP service for detecting and tracking price tags in uploaded videos."""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_DIR / "runs/train/weights/best.pt"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", PROJECT_DIR / "uploads"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_DIR / "results"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
CSV_COLUMNS = ["id", "first_frame", "confidence", "x_min", "y_min", "x_max", "y_max"]

for directory in (UPLOAD_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """Load the model only when a video is actually submitted."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def _match_tag(tags: list[dict[str, float]], x1: int, y1: int, x2: int, y2: int) -> dict[str, float] | None:
    """Return the nearest existing tag inside a centre-distance threshold."""
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    closest: dict[str, float] | None = None
    closest_distance = 100**2
    for tag in tags:
        tag_x = (tag["x1"] + tag["x2"]) / 2
        tag_y = (tag["y1"] + tag["y2"]) / 2
        distance = (center_x - tag_x) ** 2 + (center_y - tag_y) ** 2
        if distance < closest_distance:
            closest, closest_distance = tag, distance
    return closest


def process_video(video_path: Path, task_id: str) -> tuple[Path, Path, int]:
    """Run inference, write an annotated MP4 and a CSV of first detections."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open the uploaded video")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("The uploaded video has invalid frame dimensions")

        output_path = RESULTS_DIR / f"{task_id}.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Unable to create the result video")

        tracked_tags: list[dict[str, float]] = []
        csv_data: list[dict[str, float | int]] = []
        next_id = frame_number = 0
        model = get_model()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model(frame, conf=0.5, imgsz=640, verbose=False)[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, x2 = max(0, x1), min(width - 1, x2)
                    y1, y2 = max(0, y1), min(height - 1, y2)
                    confidence = float(box.conf[0])
                    tag = _match_tag(tracked_tags, x1, y1, x2, y2)
                    if tag is None:
                        tag = {"id": next_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": confidence}
                        tracked_tags.append(tag)
                        csv_data.append(
                            {
                                "id": next_id,
                                "first_frame": frame_number,
                                "confidence": round(confidence, 3),
                                "x_min": x1,
                                "y_min": y1,
                                "x_max": x2,
                                "y_max": y2,
                            }
                        )
                        next_id += 1
                    else:
                        tag.update(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"price_tag #{int(tag['id'])} {confidence:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                writer.write(frame)
                frame_number += 1
        finally:
            writer.release()

        if frame_number == 0:
            output_path.unlink(missing_ok=True)
            raise ValueError("The uploaded video contains no readable frames")
        csv_path = RESULTS_DIR / f"{task_id}.csv"
        pd.DataFrame(csv_data, columns=CSV_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8-sig")
        return output_path, csv_path, len(tracked_tags)
    finally:
        cap.release()
        video_path.unlink(missing_ok=True)


async def save_upload(video: UploadFile, destination: Path) -> None:
    """Stream the upload to disk and enforce the configured size limit."""
    written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds the allowed upload size")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await video.close()


app = FastAPI(title="Price Tag Detection", description="Video price-tag detection service")
app.mount("/static", StaticFiles(directory=PROJECT_DIR / "app/static"), name="static")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(PROJECT_DIR / "app/static/index.html")


@app.get("/dashboard", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(PROJECT_DIR / "app/static/dashboard.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model_path": str(MODEL_PATH), "model_available": MODEL_PATH.is_file()}


@app.get("/api/metrics")
async def metrics() -> dict[str, dict[str, float]]:
    return {"detection": {"precision": 0.995, "recall": 0.993, "f1_score": 0.995, "map50": 0.995}}


@app.post("/api/process")
async def process_video_api(video: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, "Supported formats: MP4, AVI, MOV, MKV")
    if not MODEL_PATH.is_file():
        raise HTTPException(503, "The detection model is unavailable on this server")

    task_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{task_id}{suffix}"
    await save_upload(video, upload_path)
    try:
        video_path, csv_path, count = await run_in_threadpool(process_video, upload_path, task_id)
        rows = pd.read_csv(csv_path).fillna("").to_dict("records")
        return {"task_id": task_id, "count": count, "results": rows, "video_ready": video_path.is_file()}
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise HTTPException(422, f"Unable to process video: {error}") from error
    except Exception as error:
        raise HTTPException(500, "Unexpected error while processing the video") from error


def result_file(task_id: str, suffix: str) -> Path:
    try:
        task_id = str(uuid.UUID(task_id))
    except ValueError as error:
        raise HTTPException(404, "Result not found") from error
    return RESULTS_DIR / f"{task_id}{suffix}"


@app.get("/api/download/{task_id}")
async def download_csv(task_id: str) -> FileResponse:
    csv_path = result_file(task_id, ".csv")
    if not csv_path.is_file():
        raise HTTPException(404, "Result not found")
    return FileResponse(csv_path, filename=f"price_tags_{task_id}.csv", media_type="text/csv")


@app.get("/api/download_video/{task_id}")
async def download_video(task_id: str) -> FileResponse:
    video_path = result_file(task_id, ".mp4")
    if not video_path.is_file():
        raise HTTPException(404, "Result not found")
    return FileResponse(video_path, filename=f"detected_{task_id}.mp4", media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
