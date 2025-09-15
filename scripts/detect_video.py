import sys
from ultralytics import YOLO
from multiprocessing import freeze_support
import os

MODEL_NAME = "train4"  # можна змінити на yolov8n / yolov8m
MODEL_PATH = os.path.join("runs", "detect", MODEL_NAME, "weights", "best.pt")

def main():
    if len(sys.argv) < 2:
        print("❌ Використання: python detect_video.py video.mp4")
        return

    video_path = sys.argv[1]

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=video_path,
        conf=0.8,
        iou=0.9,
        save=True,
        imgsz=521,
        verbose=True
    )

    # Беремо save_dir з першого результату
    save_path = results[0].save_dir if results else None

    print("✅ Обробка відео завершена")
    if save_path:
        print("Результати збережені в:", save_path)
    else:
        print("Результати не збережені")


if __name__ == "__main__":
    freeze_support()
    main()
