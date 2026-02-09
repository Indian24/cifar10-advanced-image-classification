import torch
from pathlib import Path
from src.models.baseline import BaselineCNN

CHECKPOINT_PATH = Path("results/checkpoints/best.pth")
OUTPUT_PATH = Path("results/checkpoints/model.pt")

def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "You must train the model first."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = BaselineCNN()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    dummy = torch.randn(1, 3, 32, 32)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(OUTPUT_PATH))

    print("✅ TorchScript saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
