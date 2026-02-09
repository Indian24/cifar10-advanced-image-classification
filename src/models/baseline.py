"""
Baseline model factory.
Supports resnet18 and resnet34; easy to extend.
"""

import torch.nn as nn
from torchvision import models


class BaselineModel(nn.Module):
    def __init__(self, num_classes=10, backbone="resnet18", pretrained=True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            backbone_model = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            backbone_model = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        in_features = backbone_model.fc.in_features
        backbone_model.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone_model

    def forward(self, x):
        return self.backbone(x)
