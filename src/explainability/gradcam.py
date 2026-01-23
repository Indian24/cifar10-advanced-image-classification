"""
Simple Grad-CAM implementation for CNNs (works with torchvision ResNet).
Returns heatmap numpy array and overlay plotting util.
"""

import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: nn.Module
        target_layer: the layer object (e.g., model.backbone.layer4[-1].conv2)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _save_grad(self, grad):
        self.gradients = grad

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor: torch tensor 1 x C x H x W
        class_idx: optional class index to compute gradients for
        Returns heatmap (H x W) normalized 0..1
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        self.model.zero_grad()
        score = out[0, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients  # [C, H, W] or [B, C, H, W] depending
        activations = self.activations  # [B, C, H, W]

        if grads is None or activations is None:
            raise RuntimeError("Gradients or activations are None. Ensure target layer is correct.")

        if grads.dim() == 4:
            grads = grads[0]
        if activations.dim() == 4:
            activations = activations[0]

        weights = torch.mean(grads, dim=(1, 2))  # global avg pool over H,W, shape [C]
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)  # H x W
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        heatmap = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        return heatmap
