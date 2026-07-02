
#Оценка качества обученной модели


from ultralytics import YOLO


def main():
    model = YOLO("runs/train/price_tag_detector/weights/best.pt")

    results = model.val(
        data="data/dataset_for_yolo/data.yaml",
        conf=0.25,
        iou=0.5,
    )

    print("\nРезультаты оценки:")
    print(f"   mAP50: {results.box.map50:.4f}")
    print(f"   mAP50-95: {results.box.map:.4f}")
    print(f"   Precision: {results.box.mp:.4f}")
    print(f"   Recall: {results.box.mr:.4f}")


if __name__ == "__main__":
    main()