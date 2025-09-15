import os
from ultralytics import YOLO

MODEL_NAME = "yolov8s"  # можна змінити на yolov8n / yolov8m / yolov8l

def main():
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
    # Використовуємо попередньо навчену модель
    model = YOLO(f"{MODEL_NAME}.pt")

    # Запуск тренування
    results = model.train(
        data="data.yaml",   # шлях до yaml
        epochs=50,          # кількість епох
        imgsz=768,          # розмір зображення
        batch=16,           # розмір батчу
        workers=2,          # кількість воркерів для даталоадера
        name=MODEL_NAME     # ім'я для збереженої моделі
    )

    print("✅ Тренування завершено")
    print("Модель збережено тут:", results.save_dir)


if __name__ == "__main__":
    main()
