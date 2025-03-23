import os
import random
import shutil
from pathlib import Path

# ตั้งค่า
data_dir = "data_flattened"
output_dir = "dataset"
classes = ["bottle", "can"]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# สร้างโฟลเดอร์ปลายทาง
for split in ["train", "val", "test"]:
    for cls in classes:
        Path(f"{output_dir}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

# แบ่งข้อมูล
for cls in classes:
    img_dir = Path(f"{data_dir}/{cls}")
    images = list(img_dir.glob("*.*"))  # *.jpg, *.png
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for img in train_imgs:
        shutil.copy(img, f"{output_dir}/train/{cls}/")
    for img in val_imgs:
        shutil.copy(img, f"{output_dir}/val/{cls}/")
    for img in test_imgs:
        shutil.copy(img, f"{output_dir}/test/{cls}/")

print("✅ Data split complete!")
