# Drone Detection with YOLOv8 🚁

Цей проєкт реалізує **детекцію дронів** за допомогою [Ultralytics YOLOv8](https://docs.ultralytics.com).
Модель тренується на власному датасеті та може працювати у трьох режимах:
- аналіз відео з файлу,
- обробка потоку з вебкамери у реальному часі,
- перегляд відео у реальному часі з виведенням на екран.

---

## 📂 Структура проєкту

```bash
│
├── .venv/ # віртуальне середовище Python
├── data/ # датасет (images + labels)
│ ├── images/train/
│ ├── images/val/
│ ├── labels/train/
│ ├── labels/val/
│ └── data.yaml
│
├── runs/ # результати роботи YOLO
│ └── detect/
│ ├── predict*/ # результати прогнозів (збережені зображення/відео)
│ └── <MODEL_NAME> or train*/ # результати навчання (weights, графіки)
│
├── scripts/ # корисні Python-скрипти
│ ├── 70_30.py # розділення датасету на train/val
│ ├── video_to_frames.py # розбиття відео на кадри
│ ├── train_yolo.py # тренування/дотренування моделі
│ ├── detect_video.py # аналіз відео → збереження результату
│ ├── detect_cam.py # реальний час із вебкамери
│ └── detect_video_live.py # обробка відео з виводом на екран
│ #-------------------------------------
│ # Optional (will be used in 70_30.py)
│ # The set (images & labels) can be made with https://labelstud.io/
│ # The label must be made with   <RectangleLabels ... /> for YOLO education
├── images/* # images with detected object(s)
├── labes/*  # *.txt files with coordinates
│ # end optional
│
├── yolov8n.pt # базова YOLOv8n (nano)
├── yolov8s.pt # базова YOLOv8s (small)
├── 1.mp4 # приклад відео
└── README.md
```

---
## Підняти середовище
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## ⚙️ Встановлення
```bash
git clone https://github.com/your-username/drone-detection-yolov8.git
cd obj_detection

# створення віртуального середовища
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# встановлення залежностей
pip install ultralytics opencv-python torch

```


### 🏋️ Тренування моделі

1. Поклади дані у data/images/train, data/images/val, data/labels/...

2. Запусти тренування:

   - MODEL_NAME -> yolov8n | yolov8s | yolov8m | yolo11n | yolo11s | etc...
```bash
python scripts/train_yolo.py
```
3. Модель збережеться у:
```bash
runs/detect/trainX/weights/best.pt
```

____

🔎 Використання
1. Аналіз відео та збереження результату
```
    python scripts/detect_video.py 1.mp4
```

👉 Результат збережеться у runs/detect/predictX/
___

2. Реальний час із камери
```
python scripts/detect_cam.py
```


👉 Вікно з потоком і боксами. Вихід — клавіша q.
___

3. Обробка відео у реальному часі з відображенням
```
python scripts/detect_video_live.py 1.mp4
```

👉 Кадри обробляються та показуються прямо на екрані.
___
# ⚡ Параметри

У скриптах та CLI можна налаштовувати:

- conf — поріг упевненості моделі (наприклад, 0.25–0.8)

- iou — поріг NMS (фільтрація перекриттів)

- imgsz — розмір вхідного зображення (320, 640, 1280)

# Приклад:
```
yolo detect predict model=runs/detect/train2/weights/best.pt source=1.mp4 conf=0.8 iou=0.9 imgsz=640
```

___
# 📸 Приклади результатів

(додай свої результати в docs/ і встав сюди)

Детекція на зображенні

Детекція у відео (GIF)

🚀 Плани розвитку

1. Зібрати більший датасет  зображень дронів

2. Тест моделей (yolov8m, ..., yolov8l, etc.) для підвищення точності

3. Оптимізація для AMD GPU (ONNX + DirectML)

4. Веб-інтерфейс для завантаження та аналізу відео
