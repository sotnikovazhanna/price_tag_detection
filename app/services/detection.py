
#Сервис детекции ценников


import cv2
import numpy as np
from app.services.model_service import model_service


class DetectionService:
    self.model = YOLO("runs/detect/train-2/weights/price_tag_detector.pt")

    def detect(self, frame, conf=0.5):
        results = self.model(frame, conf=conf)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(box.conf[0])
            })
        return detections


detection_service = DetectionService()