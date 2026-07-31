from ultralytics import YOLO


def main():
    print("Оценка модели YOLO")

    model = YOLO("runs/train/weights/best.pt")

    results = model.val(
        data="dataset_yolo/data.yaml",
        conf=0.5,
        iou=0.5,
    )

    print("\n Результаты оценки:")
    print(f"   mAP50: {results.box.map50:.4f}")
    print(f"   mAP50-95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall: {results.box.mr:.4f}")


if __name__ == "__main__":
    main()