import uuid
import tempfile
import cv2
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from ultralytics import YOLO

MODEL_PATH = "runs/train/weights/best.pt"
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("Загрузка модели YOLO")
model = YOLO(MODEL_PATH)
print("Модель загружена")


def process_video(video_path, task_id):
    temp_dir = Path(tempfile.mkdtemp(prefix=f"price_tag_{task_id}_"))
    temp_video_path = temp_dir / f"{task_id}_detected.mp4"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    csv_data = []

    tracked_tags = []
    next_id = 0

    print(f"\nОбработка видео: {video_path.name}")
    print(f"Всего кадров: {total_frames}, FPS: {fps:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, imgsz=640)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            matched = False
            cx_new = (x1 + x2) / 2
            cy_new = (y1 + y2) / 2

            for tag in tracked_tags:
                cx_old = (tag['x1'] + tag['x2']) / 2
                cy_old = (tag['y1'] + tag['y2']) / 2

                if abs(cx_new - cx_old) < 100 and abs(cy_new - cy_old) < 100:
                    tag['x1'] = x1
                    tag['y1'] = y1
                    tag['x2'] = x2
                    tag['y2'] = y2
                    tag['conf'] = conf
                    tag['frame'] = frame_count
                    matched = True
                    break

            if not matched:
                tracked_tags.append({
                    'id': next_id,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'conf': conf,
                    'frame': frame_count
                })
                next_id += 1

                csv_data.append({
                    'id': next_id - 1,
                    'first_frame': frame_count,
                    'confidence': round(conf, 3),
                    'x_min': int(x1),
                    'y_min': int(y1),
                    'x_max': int(x2),
                    'y_max': int(y2)
                })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"price_tag {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            progress = int((frame_count / total_frames) * 100)
            print(f"Прогресс: {progress}%", end="\r")

    cap.release()
    out.release()

    print(f"\nОбработано кадров: {frame_count}")
    print(f"Уникальных ценников: {len(tracked_tags)}")

    df = pd.DataFrame(csv_data)
    csv_path = RESULTS_DIR / f"{task_id}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    if video_path.exists():
        video_path.unlink()

    return temp_video_path, csv_path, len(tracked_tags)


def cleanup_temp_file(temp_path: Path):
    try:
        if temp_path and temp_path.exists():
            temp_path.unlink()
            if temp_path.parent.exists() and not any(temp_path.parent.iterdir()):
                temp_path.parent.rmdir()
            print(f"Удалён временный файл: {temp_path}")
    except Exception as e:
        print(f"Ошибка при удалении временного файла: {e}")


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Обнаружение ценников</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1000px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f7fa;
        }
        h1 {
            color: #1a1a2e;
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 28px;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin: 20px 0;
        }
        .upload-area {
            border: 2px dashed #c5cae9;
            padding: 50px 20px;
            text-align: center;
            border-radius: 12px;
            background: #fafafa;
        }
        .upload-area input {
            display: block;
            margin: 20px auto;
            padding: 12px;
            font-size: 16px;
        }
        .btn {
            background: #5c6bc0;
            color: white;
            padding: 14px 36px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            font-weight: 600;
        }
        .btn:hover { background: #3f51b5; }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
            color: #5c6bc0;
            font-size: 18px;
        }
        .loading .spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid #e8eaf6;
            border-top-color: #5c6bc0;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result { display: none; margin-top: 20px; }
        .result h3 { color: #1a1a2e; margin-bottom: 16px; }
        .result-links {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 16px;
        }
        .download-link {
            display: inline-block;
            background: #43a047;
            color: white;
            padding: 12px 28px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
        }
        .download-link:hover { background: #2e7d32; }
        .download-link.secondary {
            background: #1e88e5;
        }
        .download-link.secondary:hover { background: #1565c0; }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            overflow-x: auto;
            display: block;
        }
        th, td {
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid #e8eaf6;
            white-space: nowrap;
        }
        th {
            background: #f0f2ff;
            font-weight: 600;
            color: #1a1a2e;
            position: sticky;
            top: 0;
        }
        tr:hover { background: #f5f7fa; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
            margin: 16px 0;
        }
        .stat-item {
            background: #f5f7fa;
            padding: 10px 14px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-item .num { font-size: 22px; font-weight: 700; color: #1a1a2e; }
        .stat-item .label { font-size: 12px; color: #666; margin-top: 4px; }
        .error { color: #e53935; padding: 16px; background: #ffebee; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>Обнаружение ценников</h1>
    <p style="color:#666;">Загрузите видео для обнаружения ценников</p>

    <div class="card">
        <div class="upload-area">
            <div style="font-size: 48px; margin-bottom: 12px;">Видео</div>
            <p style="margin: 0 0 8px 0; font-weight: 500;">Выберите видеофайл</p>
            <p style="margin: 0 0 16px 0; font-size: 14px; color: #999;">Поддерживаются: MP4, AVI, MOV</p>
            <input type="file" id="videoFile" accept="video/mp4, video/avi, video/mov">
            <br>
            <button class="btn" id="uploadBtn" onclick="uploadVideo()">Обработать</button>
        </div>
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p style="margin-top: 16px;">Обработка видео... Это может занять несколько минут</p>
    </div>

    <div class="card result" id="result"></div>

    <script>
        const fileInput = document.getElementById('videoFile');
        const uploadBtn = document.getElementById('uploadBtn');

        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                uploadBtn.disabled = false;
            }
        });

        async function uploadVideo() {
            const file = fileInput.files[0];
            if (!file) {
                alert('Пожалуйста, выберите видеофайл');
                return;
            }

            const formData = new FormData();
            formData.append('video', file);

            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Обработка...';

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                document.getElementById('loading').style.display = 'none';
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Обработать';

                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';

                if (response.status !== 200 || data.error) {
                    resultDiv.innerHTML = `<div class="error">Ошибка: ${data.error || 'Ошибка обработки'}</div>`;
                    return;
                }

                let html = `<h3>Найдено ценников: ${data.count}</h3>`;

                if (data.results && data.results.length > 0) {
                    html += `<div class="stats">`;
                    html += `<div class="stat-item"><div class="num">${data.count}</div><div class="label">Всего детекций</div></div>`;
                    html += `</div>`;
                }

                if (data.results && data.results.length > 0) {
                    const headers = Object.keys(data.results[0]);
                    html += `<div style="overflow-x: auto; max-height: 400px; overflow-y: auto;">`;
                    html += `<table><thead><tr>`;
                    headers.forEach(h => html += `<th>${h}</th>`);
                    html += `</tr></thead><tbody>`;
                    data.results.forEach(row => {
                        html += `<tr>`;
                        headers.forEach(h => {
                            let val = row[h] || '';
                            if (typeof val === 'number') val = val.toFixed(3);
                            html += `<td>${val}</td>`;
                        });
                        html += `</tr>`;
                    });
                    html += `</tbody></table>`;
                    html += `</div>`;
                    html += `<div class="result-links">`;
                    html += `<a class="download-link" href="/api/download/${data.task_id}">Скачать CSV</a>`;
                    html += `<a class="download-link secondary" href="/api/download_video/${data.task_id}">Скачать видео</a>`;
                    html += `</div>`;
                }
                resultDiv.innerHTML = html;

            } catch (e) {
                document.getElementById('loading').style.display = 'none';
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Обработать';
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').innerHTML = `<div class="error">Ошибка: ${e.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""

app = FastAPI(title="Обнаружение ценников", description="Система обнаружения ценников для розничной торговли")

app.temp_files = {}


@app.get("/")
async def root():
    return HTMLResponse(HTML_PAGE)


@app.post("/api/process")
async def process_video_api(video: UploadFile = File(...)):
    if not video.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Поддерживаются только форматы MP4, AVI, MOV")

    task_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{task_id}.mp4"

    with open(video_path, "wb") as f:
        f.write(await video.read())

    try:
        temp_video_path, csv_path, count = process_video(video_path, task_id)

        app.temp_files[task_id] = temp_video_path

        df = pd.read_csv(csv_path)

        return {
            "task_id": task_id,
            "count": count,
            "results": df.to_dict('records')
        }
    except Exception as e:
        if video_path.exists():
            try:
                video_path.unlink()
            except Exception:
                pass
        raise HTTPException(500, f"Ошибка обработки: {str(e)}")


@app.get("/api/download/{task_id}")
async def download_csv(task_id: str):
    csv_path = RESULTS_DIR / f"{task_id}.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(
        csv_path,
        filename=f"price_tags_{task_id}.csv",
        media_type="text/csv"
    )


@app.get("/api/download_video/{task_id}")
async def download_video(task_id: str, background_tasks: BackgroundTasks):
    temp_video_path = app.temp_files.get(task_id)
    if not temp_video_path or not temp_video_path.exists():
        raise HTTPException(404, "Видео не найдено")

    background_tasks.add_task(cleanup_temp_file, temp_video_path)
    app.temp_files.pop(task_id, None)

    return FileResponse(
        temp_video_path,
        filename=f"detected_{task_id}.mp4",
        media_type="video/mp4"
    )


if __name__ == "__main__":
    import uvicorn

    print("Сервер обнаружения ценников")
    print("http://localhost:8000")
    print("Нажмите Ctrl+C для остановки")
    uvicorn.run(app, host="0.0.0.0", port=8000)



