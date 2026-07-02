
#API роуты

from fastapi import APIRouter
from app.services.metrics import metrics_collector

router = APIRouter()

@router.get("/api/metrics")
async def get_metrics():
    return metrics_collector.get_report()