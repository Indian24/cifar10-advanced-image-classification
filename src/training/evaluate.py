"""
Evaluation utilities.

Functions:
    - evaluate(model, dataloader, device, compute_loss=False)
Returns:
    val_acc (float), val_loss (float if compute_loss else None)
"""

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np


def evaluate(model, dataloader, device="cpu", compute_loss=False):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    loss_fn = nn.CrossEntropyLoss() if compute_loss else None

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            if compute_loss:
                running_loss += loss_fn(out, y).item() * x.size(0)

    acc = correct / total if total > 0 else 0.0
    avg_loss = (running_loss / total) if compute_loss else None
    model.train()
    return acc, avg_loss
