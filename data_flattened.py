import os
import shutil
from pathlib import Path

# โฟลเดอร์ต้นทางที่มีหลาย sub-folder (bottle_xxx, can_xxx)
source_dir = "raw-data"  # เปลี่ยนตามที่คุณเก็บจริง
output_dir = "data_flattened"

# สร้างโฟลเดอร์ปลายทาง
for cls in ["bottle", "can"]:
    Path(f"{output_dir}/{cls}").mkdir(parents=True, exist_ok=True)

# วนลูปดูทุกโฟลเดอร์ใน raw_data
for subfolder in os.listdir(source_dir):
    subfolder_path = os.path.join(source_dir, subfolder)

    if os.path.isdir(subfolder_path):
        # ตรวจว่าคือ bottle หรือ can
        if subfolder.startswith("bottle"):
            label = "bottle"
        elif subfolder.startswith("can"):
            label = "can"
        else:
            continue  # ข้ามโฟลเดอร์อื่น

        # คัดลอกไฟล์ทั้งหมดมาไว้ในโฟลเดอร์ที่รวม
        for filename in os.listdir(subfolder_path):
            src_file = os.path.join(subfolder_path, filename)
            if os.path.isfile(src_file):
                # เปลี่ยนชื่อไฟล์ให้ไม่ชนกัน (รวม prefix)
                new_filename = f"{subfolder}_{filename}"
                dst_file = os.path.join(output_dir, label, new_filename)
                shutil.copy(src_file, dst_file)

print("✅ รวมไฟล์ภาพเรียบร้อยแล้วไว้ที่ data_flattened/")
