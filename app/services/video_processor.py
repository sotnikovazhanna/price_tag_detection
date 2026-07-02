import cv2
import pandas as pd
from pathlib import Path

from app.config import config
from app.services.model_service import model_service


class VideoProcessor:
    """
    Обработка видео:
    - детекция + трекинг YOLO
    - сохранение обработанного видео
    - сохранение результатов в CSV
    """

    async def process(self, video_path: Path, task_id: str):

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Папка для временных видео
        temp_dir = Path(config.TEMP_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)

        output_video = temp_dir / f"{task_id}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            fps,
            (width, height)
        )

        csv_rows = []

        frame_number = 0

        print("=" * 60)
        print(f"Обработка видео: {video_path.name}")
        print(f"Размер: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Количество кадров: {total_frames}")
        print("=" * 60)

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Детекция + трекинг на КАЖДОМ кадре
            results = model_service.yolo.track(
                source=frame,
                persist=True,
                conf=0.5,
                verbose=False
            )

            result = results[0]

            # Рисуем боксы и ID
            annotated_frame = result.plot()

            # Записываем кадр в выходное видео
            writer.write(annotated_frame)

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes
                ids = boxes.id

                for i in range(len(boxes)):

                    x1, y1, x2, y2 = map(
                        int,
                        boxes.xyxy[i].cpu().numpy()
                    )

                    confidence = float(boxes.conf[i])

                    if ids is not None:
                        track_id = int(ids[i])
                    else:
                        track_id = -1

                    csv_rows.append({
                        "track_id": track_id,
                        "filename": video_path.name,
                        "frame": frame_number,
                        "timestamp_ms": int(frame_number / fps * 1000),
                        "confidence": round(confidence, 4),
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "width": x2 - x1,
                        "height": y2 - y1
                    })

            frame_number += 1

            if frame_number % 100 == 0:
                progress = frame_number / total_frames * 100
                print(f"Прогресс: {progress:.1f}%")

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        # Сохраняем CSV
        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)

        csv_path = results_dir / f"{task_id}.csv"

        df = pd.DataFrame(csv_rows)

        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("=" * 60)
        print(f"Обработка завершена")
        print(f"Найдено объектов: {len(df)}")
        print(f"CSV: {csv_path}")
        print(f"Видео: {output_video}")
        print("=" * 60)

        return df, output_video


video_processor = VideoProcessor()