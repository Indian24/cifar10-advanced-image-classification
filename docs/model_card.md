\# Model Card – CIFAR-10 Image Classifier



\## Model Overview

\- Architecture: ResNet18 (PyTorch)

\- Task: Multi-class image classification (10 classes)

\- Dataset: CIFAR-10

\- Input: 32×32 RGB images

\- Output: Class probabilities (Softmax)



\## Training Strategy

\- Two-stage training:

&nbsp; 1. Train classification head (frozen backbone)

&nbsp; 2. Fine-tune full network

\- Data augmentation: random crop, flip, optional MixUp



\## Performance

\- Stage 1 Validation Accuracy: ~92.5%

\- Stage 2 Peak Validation Accuracy: ~95.7%



\## Intended Use

\- Educational and research purposes

\- Demonstration of MLOps \& deployment pipelines



\## Limitations

\- Trained on low-resolution images

\- Not suitable for real-world safety-critical use

\- No bias analysis performed



\## Deployment

\- Exported to TorchScript

\- Served via FastAPI + Docker



