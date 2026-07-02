
#Разовое скачивание моделей

import os
import urllib.request
from paddleocr import PaddleOCR


def download_yolo():
    model_path = "models/yolov8n.pt"
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(model_path):
        print("Скачивание YOLO модели")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        urllib.request.urlretrieve(url, model_path)
        print("YOLO модель загружена")
    else:
        print("YOLO модель уже есть")


def download_paddleocr():
    print("Инициализация PaddleOCR")
    ocr = PaddleOCR(lang='ru', use_angle_cls=True, show_log=False)
    print("PaddleOCR готов")


if __name__ == "__main__":
    print("Загрузка моделей")
    download_yolo()
    download_paddleocr()
    print("Готово!")