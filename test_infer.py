import onnxruntime as ort
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# === Load ONNX model ===
onnx_model_path = "weights/efficientnet_b0.onnx"
session = ort.InferenceSession(onnx_model_path)

# === Preprocess image ===
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    return image.numpy()

# === Predict ===
def predict(image_path):
    input_tensor = preprocess(image_path)
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    probs = torch.tensor(outputs[0][0])
    conf, pred = torch.max(probs, 0)
    return pred.item(), conf.item()

# === Class names ===
class_names = ["bottle", "can"]

# === Run ===
image_path = "img/can.jpg"  # เปลี่ยนเป็น path ภาพของคุณ
pred_class, confidence = predict(image_path)
print(f"Prediction: {class_names[pred_class]}, Confidence: {confidence:.2f}")

#pip install onnx onnxruntime timm torch torchvision pillow
