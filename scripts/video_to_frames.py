import cv2
import os

# Вхідне відео
video_path = "4.mp4"
output_dir = "frames2"

# Створюємо папку для кадрів
os.makedirs(output_dir, exist_ok=True)

# Відкриваємо відео
cap = cv2.VideoCapture(video_path)

frame_count = 0
success = True

while success:
    success, frame = cap.read()
    if not success:
        break

    # Зберігаємо кадр у форматі JPG
    frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(frame_path, frame)

    frame_count += 1

cap.release()
print(f"✅ Готово! Збережено {frame_count} кадрів у папку {output_dir}")
