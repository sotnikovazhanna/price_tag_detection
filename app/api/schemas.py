
#Pydantic модели для API

from pydantic import BaseModel
from typing import Optional

class DetectionResult(BaseModel):
    filename: str
    product_name: str = "нет"
    price_default: str = "нет"
    price_card: str = "нет"
    barcode: str = "нет"
    color: str = "нет"
    frame_timestamp: int = 0
    x_min: int = 0
    y_min: int = 0
    x_max: int = 0
    y_max: int = 0

class ProcessResponse(BaseModel):
    task_id: str
    count: int
    results: list