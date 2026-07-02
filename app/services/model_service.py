
#Сервис для загрузки и хранения моделей

import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
from app.config import config

class ModelService:
    _instance = None
    _yolo = None
    _ocr = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def yolo(self):
        if self._yolo is None:
            model_path = config.MODEL_PATH
            if not os.path.exists(model_path):
                model_path = "models/yolov8n.pt"
            print(f"Загрузка YOLO: {model_path}")
            self._yolo = YOLO(model_path)
        return self._yolo

    @property
    def ocr(self):
        if self._ocr is None:
            print("Загрузка PaddleOCR")
            self._ocr = PaddleOCR(lang='ru', use_angle_cls=True, show_log=False)
        return self._ocr


model_service = ModelService()