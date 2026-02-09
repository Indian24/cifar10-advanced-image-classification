import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as T
import io
import os
import random

app = FastAPI(title="CIFAR-10 Inference API")

MODEL_PATH = "results/checkpoints/model.pth"
DEVICE = "cpu"

CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# Try loading TorchScript model, else fallback
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
        print("✅ TorchScript model loaded")
    except Exception as e:
        print("⚠️ Model load failed, using mock inference:", e)

transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor()
])

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0)

    # REAL inference if model exists
    if model:
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            idx = probs.argmax(dim=1).item()
            conf = float(probs[0][idx])
    else:
        # Mock inference (deployment demo mode)
        idx = random.randint(0, 9)
        conf = round(random.uniform(0.6, 0.95), 2)

    return {
        "class": CLASSES[idx],
        "confidence": conf,
        "mode": "real" if model else "mock"
    }
