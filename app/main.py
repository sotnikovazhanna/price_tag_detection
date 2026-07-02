import os
import uuid
import shutil
import cv2
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from ultralytics import YOLO
from starlette.background import BackgroundTask

MODEL_PATH = "runs/train/weights/best.pt"
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
TEMP_VIDEO_DIR = Path("temp_videos")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
TEMP_VIDEO_DIR.mkdir(exist_ok=True)

print("Загрузка модели YOLO...")
model = YOLO(MODEL_PATH)
print("Модель загружена.")

def remove_file(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

def process_video(video_path, task_id):
    """Обрабатывает видео через YOLO predict и сохраняет во временную папку"""

    # Запускаем YOLO predict на видео
    results = model.predict(
        source=str(video_path),
        conf=0.5,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/detect",
        name=task_id,
        exist_ok=True,
        verbose=False
    )

    # Собираем результаты в CSV
    csv_data = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                csv_data.append({
                    'filename': video_path.name,
                    'confidence': round(conf, 3),
                    'x_min': int(x1),
                    'y_min': int(y1),
                    'x_max': int(x2),
                    'y_max': int(y2)
                })

    df = pd.DataFrame(csv_data)
    csv_path = RESULTS_DIR / f"{task_id}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Ищем сохранённое видео и копируем во временную папку
    video_files = list(Path(f"runs/detect/{task_id}").glob("*.mp4"))
    temp_video_path = None

    if video_files:
        src_video = video_files[0]
        temp_video_path = TEMP_VIDEO_DIR / f"{task_id}_detected.mp4"
        shutil.copy2(src_video, temp_video_path)

        # Удаляем временную папку YOLO
        shutil.rmtree(Path(f"runs/detect/{task_id}"), ignore_errors=True)

    return df, temp_video_path


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
                    const colors = {};
                    data.results.forEach(r => {
                        const c = r.color || 'other';
                        colors[c] = (colors[c] || 0) + 1;
                    });
                    html += `<div class="stats">`;
                    const colorNames = { red: 'Красные', white: 'Белые', yellow: 'Жёлтые', other: 'Другие' };
                    for (const [key, count] of Object.entries(colors)) {
                        html += `<div class="stat-item"><div class="num">${count}</div><div class="label">${colorNames[key] || key}</div></div>`;
                    }
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
        df, temp_video_path = process_video(video_path, task_id)

        # Удаляем загруженное видео
        if video_path.exists():
            video_path.unlink()

        # Если видео создалось — сохраняем путь в словарь
        if temp_video_path and temp_video_path.exists():
            app.video_paths[task_id] = temp_video_path
        else:
            app.video_paths[task_id] = None

        return {
            "task_id": task_id,
            "count": len(df),
            "results": df.head(20).to_dict('records') if len(df) > 0 else []
        }
    except Exception as e:
        if video_path.exists():
            try:
                video_path.unlink()
            except:
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
async def download_video(task_id: str):
    video_path = app.video_paths.get(task_id)

    if not video_path or not video_path.exists():
        raise HTTPException(404, "Видео не найдено")

    return FileResponse(
        path=video_path,
        filename=f"detected_{task_id}.mp4",
        media_type="video/mp4",
        background=BackgroundTask(remove_file, video_path)
    )


# Хранилище путей к видео
app.video_paths = {}

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("Сервер обнаружения ценников")
    print("http://localhost:8000")
    print("Нажмите Ctrl+C для остановки")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)