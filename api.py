from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import os
import shutil
import requests
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
from io import BytesIO

app = FastAPI()

# === Load EfficientNetB0 Model (PyTorch) ===
NUM_CLASSES = 2  # bottle / can
class_names = ["bottle", "can"]

def load_effnet_model(path):
    model = efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    # model.load_state_dict(torch.load(path, map_location='cpu'))  # <- For state_dict only
    state_dict = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)

    model.eval()
    return model

model_effnet = load_effnet_model('weights/efficientnet_b0_3class.pth')

transform_effnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Bottle & Can Models (TensorFlow) ===
model_bottle_size = tf.keras.models.load_model('weights/bottle_size_model.h5')
model_bottle_brand = tf.keras.models.load_model('weights/bottle_brand_model.h5')
model_can_size = tf.keras.models.load_model('weights/can_size_model.h5')
model_can_brand = tf.keras.models.load_model('weights/can_brand_model.h5')

bottle_size_classes = ['bottel_1600', 'bottel_350', 'bottle_1250', 'bottle_1500', 'bottle_1950', 'bottle_280', 'bottle_300', 'bottle_320', 'bottle_322', 'bottle_340', 'bottle_360', 'bottle_400', 'bottle_410', 'bottle_430', 'bottle_440', 'bottle_445', 'bottle_500', 'bottle_600ml', 'bottle_640', 'bottle_750']
bottle_brand_classes = ['amphawa', 'amwelplus', 'aquafina', 'beauti_drink', 'big', 'coca_cola', 'cocomax', 'crystal', 'est', 'ichitan', 'kato', 'mansome', 'mikko', 'minearlwater', 'nestle', 'no_band', 'oishi', 'pepsi', 'sing', 'spinking_water', 'sprite', 'srithep', 'tipchumporn_drinking_water']
can_size_classes = ['can_180', 'can_245', 'can_330', 'can_490']
can_brand_classes = ['birdy', 'calpis_lacto', 'chang', 'green_mate', 'leo', 'nescafe', 'sing']

CONFIDENCE_THRESHOLD = 0.6
API_URL = "https://cookkeptback.sehub-thailand.com"

def upload_image_to_strapi(image_data, filename):
    url = f"{API_URL}/api/upload"
    files = {'files': (filename, image_data, 'image/jpeg')}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        print(f"Uploaded {filename} to Strapi successfully.")
        return response.json()
    else:
        print(f"Failed to upload {filename} to Strapi: {response.status_code}, {response.text}")
        raise HTTPException(status_code=500, detail="Failed to upload image to Strapi")

def preprocess_image_tf(image_bytes):
    image = Image.open(BytesIO(image_bytes)).resize((256, 144))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict_category(image_bytes, model_size, model_brand, size_classes, brand_classes):
    processed_image = preprocess_image_tf(image_bytes)
    size_prediction = model_size.predict(processed_image)
    size_index = np.argmax(size_prediction)
    size_confidence = np.max(size_prediction)

    brand_prediction = model_brand.predict(processed_image)
    brand_index = np.argmax(brand_prediction)
    brand_confidence = np.max(brand_prediction)

    size_name = size_classes[size_index] if size_confidence >= CONFIDENCE_THRESHOLD else "unknown"
    brand_name = brand_classes[brand_index] if brand_confidence >= CONFIDENCE_THRESHOLD else "unknown"

    return size_name, brand_name

def classify_item_with_effnet(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = transform_effnet(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_effnet(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, predicted_idx = torch.max(probs, dim=0)
        predicted_class = class_names[predicted_idx.item()]
    return predicted_class, conf.item()

def save_image(image_data, path_image):
    with open(path_image, "wb") as f:
        f.write(image_data)

def delete_image(path_image):
    if os.path.exists(path_image):
        os.remove(path_image)

def model_process(file, image_data, expected_item):
    path_image = f"images/{expected_item}/valid/{file.filename}"
    upload_image_to_strapi(image_data, file.filename)
    save_image(image_data, path_image)
    print(f"Saved image to {path_image}")
    predicted_item, confidence = classify_item_with_effnet(image_data)
    print(f"Predicted: {predicted_item}, Confidence: {confidence}")
    delete_image(path_image)

    if predicted_item != expected_item:
        raise HTTPException(status_code=400, detail={"error": f"{expected_item} not detected. Got: {predicted_item}"})

    return predicted_item, confidence

@app.post("/processImageBottle")
async def process_image_bottle(file: UploadFile = File(...)):
    try:
        item = "bottle"
        image_data = await file.read()
        predicted_class, confidence = model_process(file, image_data, item)
        size, brand = predict_category(image_data, model_bottle_size, model_bottle_brand, bottle_size_classes, bottle_brand_classes)
        return {
            "isValidBottle": predicted_class == item,
            "brand": brand,
            "size": size,
            "object_confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": f"Invalid image format: {str(e)}"})

@app.post("/processImageCan")
async def process_image_can(file: UploadFile = File(...)):
    try:
        item = "can"
        image_data = await file.read()
        predicted_class, confidence = model_process(file, image_data, item)
        size, brand = predict_category(image_data, model_can_size, model_can_brand, can_size_classes, can_brand_classes)
        return {
            "isValidCan": predicted_class == item,
            "brand": brand,
            "size": size,
            "object_confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": f"Invalid image format: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run: uvicorn main:app --reload --port 8000
