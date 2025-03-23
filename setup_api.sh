#!/bin/bash

# สร้าง Virtual Environment
# python3 -m venv venv
python3.11 -m venv venv # for use tensorflow

# เปิดใช้งาน Virtual Environment
source ./venv/bin/activate

echo "📦 Installing dependencies..."

# ติดตั้ง dependencies ที่จำเป็น
# FastAPI & Uvicorn
pip install fastapi uvicorn

# Image & Data processing
pip install pillow requests numpy

# PyTorch (CPU version)
pip install torch torchvision

# TensorFlow (regular)
pip install tensorflow-macos
pip install tensorflow-metal


# Machine learning utilities
pip install scikit-learn

echo "✅ All dependencies installed successfully!"
