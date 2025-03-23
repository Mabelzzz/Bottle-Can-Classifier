import torch
import timm
import torch.nn as nn
import os

# === CONFIG ===
MODEL_PATH = "weights/efficientnet_model.pth"
ONNX_EXPORT_PATH = "weights/efficientnet_b0.onnx"
NUM_CLASSES = 2  # bottle / can
INPUT_SIZE = (3, 224, 224)

# === Load Model ===
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# === Dummy input ===
dummy_input = torch.randn(1, *INPUT_SIZE)

# === Export ===
torch.onnx.export(
    model, dummy_input, ONNX_EXPORT_PATH,
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ Exported ONNX model to: {ONNX_EXPORT_PATH}")

#pip install onnx onnxruntime timm torch torchvision pillow
