import torch
import torch.nn as nn
from fastapi import FastAPI

app = FastAPI(title="CIFAR-10 Inference API")

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

MODEL = DummyModel()
MODEL.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    return {"class": "cat", "confidence": 0.42}


@app.get("/health")
def health():
    return {"status": "ok"}
