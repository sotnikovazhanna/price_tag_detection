
#Упрощённый трекинг ценников


class TrackingService:

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.tracks = []

    def update(self, detections: list):
        new_tracks = []
        for det in detections:
            matched = False
            for track in self.tracks:
                if self._compute_iou(det['bbox'], track['bbox']) > self.iou_threshold:
                    track['bbox'] = det['bbox']
                    track['confidence'] = det['confidence']
                    track['frames_seen'] += 1
                    matched = True
                    break
            if not matched:
                new_tracks.append({
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'frames_seen': 1
                })

        self.tracks = new_tracks
        return self.tracks

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0


tracking_service = TrackingService()