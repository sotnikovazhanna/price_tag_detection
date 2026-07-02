
#Сбор и расчёт метрик


import json
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class DetectionMetrics:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self):
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self):
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self):
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


class MetricsCollector:
    def __init__(self):
        self.detection = DetectionMetrics()

    def get_report(self):
        return {
            'detection': {
                'precision': round(self.detection.precision, 4),
                'recall': round(self.detection.recall, 4),
                'f1_score': round(self.detection.f1, 4)
            },
            'timestamp': datetime.now().isoformat()
        }


metrics_collector = MetricsCollector()