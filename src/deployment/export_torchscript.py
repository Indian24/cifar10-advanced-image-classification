import torch
import os
from src.models.baseline import BaselineCNN

CHECKPOINT = "results/checkpoints/best.pth"
OUT = "results/checkpoints/model.pt"

def load_model():
    model = BaselineCNN()
    ckpt = torch.load(CHECKPOINT, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model

if __name__ == "__main__":
    os.makedirs("results/checkpoints", exist_ok=True)
    model = load_model()
    dummy = torch.randn(1, 3, 32, 32)
    traced = torch.jit.trace(model, dummy)
    traced.save(OUT)
    print("Saved TorchScript model to:", OUT)

