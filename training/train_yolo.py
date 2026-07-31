from ultralytics import YOLO


def main():
    print("Обучение YOLO для детекции ценников")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data="dataset_yolo/data.yaml",      # путь к датасету
        epochs=50,                          # количество эпох
        imgsz=640,                          # размер изображений
        batch=8,                            # размер батча
        device="cpu",                       # или "cuda" для GPU
        workers=0,                          # для Windows
        patience=10,                        # ранняя остановка
        project="runs/train",               # куда сохранять
        name="price_tag_detector",          # имя эксперимента
        exist_ok=True,                      # перезаписывать
        # Аугментация
        mosaic=0.5,
        mixup=0.3,
    )

    print("\n Обучение завершено!")
    print(f" Результаты: runs/train/price_tag_detector")
    print(f" Лучшая модель: runs/train/price_tag_detector/weights/best.pt")


if __name__ == "__main__":
    main()