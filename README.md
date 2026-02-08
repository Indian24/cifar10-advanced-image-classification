# CIFAR-10 Advanced Image Classification — Multi-Level Deep Learning System

[![CI](https://img.shields.io/badge/ci-%20placeholder-lightgrey)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

**One-line:** Production-oriented CIFAR-10 image classification project demonstrating a progressive pipeline from transfer-learning baseline to research-grade, deployment-ready models (two-stage training, MixUp, Grad-CAM, ensembles).

## Highlights
- Two-stage training (head-only → end-to-end fine-tune)
- MixUp augmentation + advanced augmentation pipeline
- Interpretability: Grad-CAM saliency maps
- Ensemble support (hard/soft voting)
- Colab notebooks for reproducible experiments
- Configured via `configs/training.yaml` for hyperparameters & paths


## 🚀 Quickstart (Inference)

Build and run the inference API using Docker:

```bash
docker build -t cifar10-infer .
docker run -p 8080:8080 cifar10-infer

---

## Table of contents
1. [Project Overview](#project-overview)  
2. [Why this project / Who should use this](#why-this-project--who-should-use)  
3. [Dataset summary](#dataset-summary)  
4. [Level-wise challenge (L1 → L4)](#level-wise-challenge-l1-—-l4)  
5. [Quickstart (local & Colab)](#quickstart-local--colab)  
6. [Install & Dependencies](#install--dependencies)  
7. [How to run (train / eval / infer)](#how-to-run-train--eval--infer)  
8. [Project structure & key files](#project-structure--key-files)  
9. [Examples & expected outputs](#examples--expected-outputs)  
10. [Experiment reproducibility](#experiment-reproducibility)  
11. [Results (reported)](#results-reported)  
12. [Visuals & artifacts](#visuals--artifacts)  
13. [How to contribute](#how-to-contribute)  
14. [License](#license)  
15. [Contact / Authors](#contact--authors)  
16. [References](#references)  
17. [Notes / Caveats](#notes--caveats)

---

## Project overview
This repository contains code, configs and Colab notebooks to train and evaluate image classification models on CIFAR-10. The goal is to present an industry-grade workflow covering data pipelines, model training (transfer learning, two-stage training, ensembles), interpretability, experiment logging, and reproducibility.

The project emphasizes:
- Clean configuration via `configs/*.yaml`
- Reproducible experiments (seeded runs, deterministic data splits)
- Analysis artifacts: plots, per-class metrics, Grad-CAM images, and a research-style report

---

## Why this project / Who should use this
**Why:** Serves as a compact reference for building production-ready image classification pipelines and for demonstrating ML engineering best practices in interviews, hiring challenges, and technical demos.

**Who should use:** ML engineers, researchers, students preparing portfolios, and hiring managers evaluating applied ML skills.

---

## Dataset summary
- **Dataset:** CIFAR-10  
- **Total images:** 60,000 (32×32 RGB)  
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
- **Split:** train = 50,000, test = 10,000  
- **Source / reference:** TensorFlow Datasets / Torchvision auto-download.  
  TensorFlow listing: https://www.tensorflow.org/datasets/catalog/cifar10

---

## Level-wise challenge (L1 → L4)

### Level 1 — Baseline Model
- **Objective:** Transfer-learning baseline (e.g., ResNet50/ResNet18).  
- **Approach:** Pre-trained CNN, fine-tune classifier head for CIFAR-10.  
- **Expected accuracy:** ≥ **85%**  
- **Deliverables:** data loader, baseline model, test accuracy, training/validation curves  
- **Colab:** https://colab.research.google.com/drive/1MBBx5bhc4_OFgHj7oVW5FCNlD-yiSq46?usp=sharing

---

### Level 2 — Intermediate Techniques
- **Objective:** Improve baseline using augmentation, regularization, tuning.  
- **Approach:** data augmentation (random crop, flip), MixUp, L2/weight decay, two-stage training.  
- **Expected accuracy:** ≥ **90%**  
- **Deliverables:** augmentation pipeline, ablation study, accuracy comparison, analysis report  
- **Reported / example run:**  
  - **Stage-1 (head training w/ strong aug + MixUp):** validation ~**92.5%**  
  - **Stage-2 (fine-tune end-to-end with lighter aug + lower LR):** peak validation **95.7%**  
- **Colab:** https://colab.research.google.com/drive/1SGw96OxxcLhKfvxllA4AKFo8RH49bbqC?usp=sharing

---

### Level 3 — Advanced Architecture Design
- **Objective:** Custom/advanced architectures (attention, improved CNN).  
- **Expected accuracy:** ≥ **91%**  
- **Deliverables:** architecture explanation, implementation, per-class accuracy & confusion matrix, Grad-CAM analysis.  
- **Colab:** https://colab.research.google.com/drive/1EpSJs8627GmuKOln0sbies6deRyDLnYg?usp=sharing

---

### Level 4 — Expert Techniques
- **Objective:** Near SOTA performance via ensembles, meta-learning, or novel strategies.  
- **Expected accuracy:** ≥ **93%**  
- **Deliverables:** multiple trained models, ensemble strategy, comparative analysis, research-quality report (~10 pages).  
- **Colab:** https://colab.research.google.com/drive/12-LbKhmL7SYjYeTtdz1QNRp4FEyBwoHD?usp=sharing

---

## Quickstart (local and Colab)

### Clone repository (local)
```bash
git clone https://github.com/Indian24/cifar10-advanced-image-classification.git
cd cifar10-advanced-image-classification


Open in Google Colab

Open any of the Colab links above and run the notebook cells. Colab will run the experiments and auto-download CIFAR-10.


Install & dependencies

Recommended: Python 3.8+ and a virtual environment (conda / venv).

# create env (conda example)
conda create -n cifar10 python=3.8 -y
conda activate cifar10

# install core requirements
pip install -r requirements.txt
# or minimal
pip install torch torchvision numpy matplotlib seaborn


requirements.txt contains pinned or recommended versions. Hyperparameters and paths are configured in configs/training.yaml.



How to run (train / evaluate / infer)

Train (local) — configurable through configs/training.yaml


# run default training config
python src/training/train.py --config configs/training.yaml


Evaluate

python src/training/evaluate.py --checkpoint results/checkpoints/best.pth



Inference (example)

# quick inference example (script)
python src/deployment/infer.py --checkpoint results/checkpoints/best.pth --image samples/airplane.png

Notes: All scripts read default paths & hyperparams from configs/training.yaml. Overridable via CLI args in most scripts.



Project structure & key files

cifar10-advanced-image-classification/
├── configs/
│   └── training.yaml          # main training hyperparameters & paths
├── experiments/               # Colab/Notebook experiments (Level1..4)
├── src/
│   ├── data/
│   │   └── dataset.py         # dataloaders, augmentations
│   ├── models/
│   │   └── baseline.py        # baseline model wrappers (ResNet etc.)
│   ├── training/
│   │   ├── train.py           # training loop (two-stage support)
│   │   └── evaluate.py        # evaluation utilities
│   └── deployment/
│       └── infer.py           # inference helper (optional)
├── results/                   # logs, tensorboard, checkpoints
├── requirements.txt
└── README.md






Key files

configs/training.yaml — default hyperparameters (epochs, lr, batch_size, weight_decay, two_stage flag). Edit to reproduce different runs.

src/training/train.py — training loop (supports MixUp, two-stage, checkpointing).

src/data/dataset.py — transforms and dataloaders (auto-download CIFAR-10).

experiments/*.ipynb — guided Colab notebooks for each level / ablation.






Examples & expected outputs

Start training (example)

python src/training/train.py --config configs/training.yaml



Typical console output (example):


Stage 1: training head only.
Epoch [1/12], train_loss: 1.9093, train_acc: 0.3448, val_acc: 0.3516, time: 85.0s
...
Epoch [5/12], train_loss: 0.3452, train_acc: 0.9152, val_acc: 0.9250, time: 82.1s
Stage 2: unfreezing all parameters (fine-tune).
...
Final test accuracy: 0.957




Evaluate

python src/training/evaluate.py --checkpoint results/checkpoints/best.pth
# prints metrics: accuracy, per-class precision/recall, confusion matrix



Experiment reproducibility

Default config

configs/training.yaml (defaults in repo):


train:
  epochs: 12
  batch_size: 128
  num_workers: 4
  lr: 0.01
  momentum: 0.9
  weight_decay: 1e-4
  val_split: 0.1
  two_stage: true
  two_stage_epochs: 3
model:
  backbone: resnet18
  num_classes: 10
  pretrained: true
augmentation:
  enabled: true
  random_crop: true
  horizontal_flip: true
  mixup_alpha: 0.0





Seeds & deterministic

Set seed in scripts: seed = 42 (configurable in configs/training.yaml).

For full determinism set torch.backends.cudnn.deterministic = True and torch.backends.cudnn.benchmark = False in train.py (note: may affect performance).

Two-stage training summary

Stage-1: Freeze backbone, train head with strong augmentations and MixUp (reported val ≈ 92.5%).

Stage-2: Unfreeze and fine-tune end-to-end with lighter augmentation and reduced LR → peak validation 95.7%.

Reproducibility checklist



# 1) clone
git clone https://github.com/Indian24/cifar10-advanced-image-classification.git
cd cifar10-advanced-image-classification

# 2) setup env & deps
conda create -n cifar10 python=3.8 -y
conda activate cifar10
pip install -r requirements.txt

# 3) run training
python src/training/train.py --config configs/training.yaml

# 4) evaluate best checkpoint
python src/training/evaluate.py --checkpoint results/checkpoints/best.pth




Open Colab notebooks

Level 1 notebook: open https://colab.research.google.com/drive/1MBBx5bhc4_OFgHj7oVW5FCNlD-yiSq46?usp=sharing

Level 2 notebook: open https://colab.research.google.com/drive/1SGw96OxxcLhKfvxllA4AKFo8RH49bbqC?usp=sharing

Level 3 notebook: open https://colab.research.google.com/drive/1EpSJs8627GmuKOln0sbies6deRyDLnYg?usp=sharing

Level 4 notebook: open https://colab.research.google.com/drive/12-LbKhmL7SYjYeTtdz1QNRp4FEyBwoHD?usp=sharing





Results (reported / example)

| Level                   | Target Accuracy | Notes / Reported                              |
| ----------------------- | --------------: | --------------------------------------------- |
| Level 1 (baseline)      |           ≥ 85% | Transfer-learning baseline                    |
| Level 2 (intermediate)  |           ≥ 90% | Stage-1 ≈ **92.5%**, Stage-2 peak **95.7%**   |
| Level 3 (advanced arch) |           ≥ 91% | Custom architecture + Grad-CAM analysis       |
| Level 4 (expert)        |           ≥ 93% | Ensemble / meta strategies (research quality) |

Use these numbers as expected references — actual results depend on hyperparameters and compute.






Visuals & artifacts

Add images / diagrams to docs/ and reference them in notebooks and README:

docs/arch_diagram.png — architecture diagram placeholder

docs/gradcam_class_dog.png — Grad-CAM example

results/plots/train_val_curves.png — training/validation loss & accuracy curves
Guidance: Save generated figures to results/figures/ and link in README or notebook outputs.





How to contribute

Fork → branch from main (feature/<name>).

Implement changes & add tests / notebooks.

Keep configs/ changes documented.

Create PR with clear description, experiment logs, and artifacts.

Use issue templates and link any reproduce scripts or Colab notebooks.

Suggested PR checklist:

 Code follows style & linting

 Configs updated or documented

 Notebooks runnable & outputs reproducible

 Results / plots checked into results/ (if applicable)




 License

This project is provided under the MIT License — see LICENSE.






Contact / Authors

Author: Indian24 (GitHub: Indian24
)

For questions, issues, or collaboration requests open an issue or PR on the repository.




References

CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

Torchvision datasets: https://pytorch.org/vision/stable/datasets.html

Grad-CAM paper & resources





Notes / Caveats

Auto-download: CIFAR-10 is auto-downloaded by Torchvision when download=True. If running behind a firewall, manually place the dataset under ./data/ or change configs/training.yaml root path.

GPU: Recommended for training. Use Colab GPU for quick experiments.

Windows CRLF warning: Notebooks and scripts may show LF/CRLF warnings on Windows — acceptable but consider normalizing line endings before committing.

Determinism vs performance: For deterministic reproducibility set cuDNN deterministic flags (may slow training).





Appendix — Useful commands

# run a quick smoke test (one epoch, small batch)
python src/training/train.py --config configs/training.yaml --dry_run True

# list tracked files & current git status
git status
git add .
git commit -m "experiment: two-stage run"
git push origin main




Thank you — contributions, issues, and improvements are welcome.

