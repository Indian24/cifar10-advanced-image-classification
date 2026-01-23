"""
Train script.

Usage:
    python -m src.training.train --config configs/training.yaml

Features:
- reads YAML config
- builds dataloaders
- builds model (resnet18 default)
- two-stage training (head-only then fine-tune) if configured
- checkpointing (best val accuracy)
- optional MixUp (alpha in config)
- TensorBoard logging
"""

import argparse
import os
import yaml
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
# TensorBoard disabled (Windows fix)

from src.data.dataset import get_dataloaders
from src.models.baseline import BaselineModel
from src.training.evaluate import evaluate  # will return val accuracy and confusion if needed


def mixup_data(x, y, alpha=1.0, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_path)


def train_loop(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", True) else "cpu")
    batch_size = cfg["train"].get("batch_size", 128)
    num_workers = cfg["train"].get("num_workers", 4)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
    batch_size=batch_size,
    augment=cfg.get("augmentation", {}).get("enabled", True),
    augment_params=cfg.get("augmentation", {}),
    val_split=cfg["train"].get("val_split", 0.1),
    num_workers=num_workers,
    download=True
)


    model_cfg = cfg.get("model", {})
    model = BaselineModel(num_classes=model_cfg.get("num_classes", 10),
                          backbone=model_cfg.get("backbone", "resnet18"),
                          pretrained=model_cfg.get("pretrained", True))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(cfg["train"].get("lr", 0.01)),
                          momentum=float(cfg["train"].get("momentum", 0.9)),
                          weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg["train"].get("lr_step", 30),
                                          gamma=cfg["train"].get("lr_gamma", 0.1))

    start_epoch = 0
    best_val = 0.0

    # TensorBoard
    tb_dir = None
    # tensorboard disabled
    writer = None

    # Optional two-stage training (train head then finetune)
    two_stage = cfg.get("train", {}).get("two_stage", False)
    if two_stage:
        # Freeze backbone parameters, train head only
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False
        print("Stage 1: training head only.")

    mixup_alpha = cfg.get("augmentation", {}).get("mixup_alpha", 0.0)

    for epoch in range(start_epoch, cfg["train"].get("epochs", 10)):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        tic = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # MixUp
            if mixup_alpha and mixup_alpha > 0:
                import numpy as np  # local import
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha, device=device)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                preds = outputs.argmax(dim=1)
                # For accuracy logging we compute against targets_a only (approx)
                batch_correct = (preds == targets_a).sum().item()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                batch_correct = (preds == targets).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += batch_correct
            running_total += inputs.size(0)

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        scheduler.step()

        # Validation
        val_acc, val_loss = evaluate(model, val_loader, device=device, compute_loss=True)
        # tensorboard disabled
        # tensorboard disabled
        # tensorboard disabled
        # tensorboard disabled

        print(f"Epoch [{epoch+1}/{cfg['train'].get('epochs')}], train_loss: {epoch_loss:.4f}, train_acc: {epoch_acc:.4f}, val_acc: {val_acc:.4f}, time: {time.time()-tic:.1f}s")

        is_best = val_acc > best_val
        if is_best:
            best_val = val_acc

        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val
        }, is_best=is_best, checkpoint_dir=cfg.get("training", {}).get("checkpoint_dir", "results/checkpoints"))

        # After stage 1, unfreeze for stage 2
        if two_stage and epoch + 1 == cfg["train"].get("two_stage_epochs", 5):
            for p in model.parameters():
                p.requires_grad = True
            print("Stage 2: unfreezing all parameters (fine-tune).")

    # tensorboard disabled
    # final test
    test_acc, _ = evaluate(model, test_loader, device=device, compute_loss=True)
    print("Final test accuracy:", test_acc)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_loop(cfg)
