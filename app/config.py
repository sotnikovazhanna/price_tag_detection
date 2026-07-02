import os

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
    UPLOAD_DIR = "uploads"
    RESULTS_DIR = "results"
    DB_PATH = "data/products.db"
    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
    FRAME_SKIP = 15
    TEMP_DIR = "temp"

config = Config()