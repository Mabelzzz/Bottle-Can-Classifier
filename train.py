import torch
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("dataset_obj/train", transform=transform)
val_data = datasets.ImageFolder("dataset_obj/val", transform=transform)
test_data = datasets.ImageFolder("dataset_obj/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# 2. Load EfficientNetB0 (Fix deprecated pretrained param)
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)  # <- NEW
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # 2 classes: bottle / can
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 3. Train
for epoch in range(100):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

# 4. Evaluate
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.cpu().tolist())

acc = accuracy_score(y_true, y_pred)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# 5. Save PyTorch model (state_dict only)
torch.save(model.state_dict(), "weights/efficientnet_b0_3class.pth")

# 6. Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model, dummy_input, "weights/efficientnet_b0_3class.onnx",
    input_names=["input"], output_names=["output"],
    export_params=True, opset_version=11
)

print("🚀 Model saved as weights/efficientnet_b0_3class.pth and .onnx")
