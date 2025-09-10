import sys
from ultralytics import YOLO

# Шлях до моделі
MODEL_PATH = "runs/detect/train2/weights/best.pt"

def main():
    if len(sys.argv) < 2:
        print("❌ Використання: python detect_video.py video.mp4")
        return

    video_path = sys.argv[1]
    model = YOLO(MODEL_PATH)

    # Аналіз відео і збереження результату (AVI)
    model.predict(
        source=video_path,
        conf=0.8,
        iou=0.9,
        save=True,         # збереження результату
        imgsz=521
    )

if __name__ == "__main__":
    main()
