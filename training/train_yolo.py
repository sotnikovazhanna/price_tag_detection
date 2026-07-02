
#Дообучение YOLO для распознавания ценников


from ultralytics import YOLO
import torch


def main():
    print("ДООБУЧЕНИЕ YOLO ДЛЯ ЦЕННИКОВ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Device: {device}")

    model = YOLO("models/yolov8n.pt")

    results = model.train(
        data="data/dataset_for_yolo/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        patience=10,
        save=True,
        project="runs/train",
        name="price_tag_detector",
        exist_ok=True,
        mosaic=0.5,
        mixup=0.3,
    )

    print("\nОбучение завершено")
    print("Результаты: runs/train/price_tag_detector")
    print("Лучшая модель: runs/train/price_tag_detector/weights/best.pt")


if __name__ == "__main__":
    main()