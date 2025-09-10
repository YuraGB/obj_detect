import os
import random
import shutil

# Вхідні папки
images_dir = "images"
labels_dir = "labels"

# Вихідні папки
base_dir = "data"
train_img_dir = os.path.join(base_dir, "images/train")
val_img_dir = os.path.join(base_dir, "images/val")
train_lbl_dir = os.path.join(base_dir, "labels/train")
val_lbl_dir = os.path.join(base_dir, "labels/val")

# Створюємо папки
for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Беремо всі картинки
images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
random.shuffle(images)

split_idx = int(len(images) * 0.7)  # 70% train
train_files = images[:split_idx]
val_files = images[split_idx:]

def move_files(files, target_img_dir, target_lbl_dir):
    for img_file in files:
        lbl_file = img_file.replace(".jpg", ".txt")
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(target_img_dir, img_file))
        shutil.copy(os.path.join(labels_dir, lbl_file), os.path.join(target_lbl_dir, lbl_file))

move_files(train_files, train_img_dir, train_lbl_dir)
move_files(val_files, val_img_dir, val_lbl_dir)

print(f"✅ Train: {len(train_files)} | Val: {len(val_files)}")
