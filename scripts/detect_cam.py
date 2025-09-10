import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = "runs/detect/train2/weights/best.pt"

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_half = (device != "cpu")  # FP16 лише на CUDA

    model = YOLO(MODEL_PATH).to(device)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Камера не відкрита")
        return

    while True:
        # Скидаємо застарілі кадри
        for _ in range(2):
            cap.grab()

        ok, frame = cap.read()
        if not ok:
            break

        # Прогноз без збереження, але швидко
        results = model.predict(
            source=frame,
            conf=0.8,
            iou=0.9,
            imgsz=416,
            device=device,
            half=use_half,
            verbose=False
        )

        annotated = results[0].plot()
        cv2.imshow("Real-time Camera Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
