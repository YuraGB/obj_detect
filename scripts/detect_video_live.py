import sys
import cv2
import torch
from ultralytics import YOLO
import time

MODEL_PATH = "runs/detect/train2/weights/best.pt"

def main():
    if len(sys.argv) < 2:
        print("❌ Використання: python detect_video_live.py video.mp4")
        return

    video_path = sys.argv[1]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_half = (device != "cpu")

    model = YOLO(MODEL_PATH).to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Відео не відкрите")
        return

    # Зменшимо розмір для швидкості
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()

        # Інференс на меншому розмірі
        results = model.predict(
            source=frame,
            conf=0.6,       # можна знизити для швидкості
            iou=0.7,
            imgsz=320,      # менше = швидше
            device=device,
            half=use_half,
            verbose=False
        )

        annotated = results[0].plot()

        dt = (time.time() - t0) * 1000
        cv2.putText(annotated, f"{dt:.1f} ms ({1000/dt:.1f} FPS)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Video Live Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
