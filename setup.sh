#!/bin/bash

# สร้าง Virtual Environment
python3 -m venv venv
# python3.11 -m venv venv # for use tensorflow

# เปิดใช้งาน Virtual Environment
source ./venv/bin/activate
echo "📦 Installing Python dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn pillow scikit-learn onnx

echo "✅ Setup complete!"
