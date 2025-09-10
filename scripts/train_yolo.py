import os
from ultralytics import YOLO

# === 1. Створюємо data.yaml ===
yaml_content = """train: ./data/images/train
val: ./data/images/val

nc: 1
names: ['drone']
"""

os.makedirs("data", exist_ok=True)
with open("data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("✅ data.yaml створено")

# === 2. Тренування YOLO ===
# Використовуємо yolov8n (Nano) для швидкості, можна замінити на yolov8s / yolov8m
model = YOLO("yolov8n.pt")

# Запуск тренування
results = model.train(
    data="data.yaml",  # шлях до yaml
    epochs=50,                 # кількість епох
    imgsz=640,                 # розмір зображення
    batch=16,                  # розмір батчу (змінюй залежно від RAM/GPU)
    workers=2,                 # кількість воркерів для даталоадера
)

print("✅ Тренування завершено")
print("Модель збережено тут:", results.save_dir)
