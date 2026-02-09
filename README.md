# CIFAR-10 Advanced Image Classification — Multi-Level Deep Learning System

[![CI](https://img.shields.io/badge/ci-%20placeholder-lightgrey)](#) [![License](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

**One-line:** Production-oriented CIFAR-10 image classification repo demonstrating a progressive pipeline from transfer-learning baseline to research-grade, deployment-ready models (two-stage training, MixUp, Grad-CAM, ensembles) with cloud/DevOps readiness for hiring assessments.

---

## Highlights

* Two-stage training (head-only → end-to-end fine-tune)
* MixUp + advanced augmentation pipeline
* Interpretability: Grad-CAM visualizations
* Export to ONNX / TorchScript, FastAPI + Docker inference
* CI (GitHub Actions) smoke tests + container build example
* Experiment tracking (W&B / MLflow patterns) and reproducibility
* Deployment playbook: Docker → Cloud Run / ECS / Kubernetes

---

## Table of contents

* [Project overview](#project-overview)
* [Why this project / Who should use this](#why-this-project--who-should-use)
* [Dataset summary](#dataset-summary)
* [Level-wise challenge (L1 → L4)](#level-wise-challenge-l1-—-l4)
* [Quickstart (local / Colab / inference)](#quickstart-local--colab--inference)
* [Install & dependencies](#install--dependencies)
* [How to run (train / evaluate / infer)](#how-to-run-train--evaluate--infer)
* [Project structure & key files](#project-structure--key-files)
* [Examples & expected outputs](#examples--expected-outputs)
* [Experiment reproducibility & two-stage summary](#experiment-reproducibility--two-stage-summary)
* [Results (reported)](#results-reported)
* [Model card & limitations](#model-card--limitations)
* [Deployment & MLOps playbook](#deployment--mlops-playbook)
* [How to contribute](#how-to-contribute)
* [Resume bullets & interview talking points](#resume-bullets--interview-talking-points)
* [License & contact](#license--contact)
* [Notes / caveats](#notes--caveats)

---

## Project overview

This repository is a production-focused ML project implementing CIFAR-10 classification with strong emphasis on: clear configs, reproducible experiments, model explainability, and end-to-end deployment readiness. It is designed to demonstrate both ML research skills and practical engineering (Docker, CI, export formats, deployment).

**Goal for hiring assessments:** provide a single portfolio project that proves you can design models, reproduce experiments, package the model into a service, and deliver a cloud-aware deployment playbook.

---

## Why this project / Who should use this

**Why:** compact, complete demo of ML + MLOps best practices.
**Who:** ML/ML-Engineering candidates, early-career ML engineers, students preparing technical interviews or hiring challenge submissions.

---

## Dataset summary

* **Dataset:** CIFAR-10
* **Images:** 60,000 (32×32 RGB)
* **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
* **Split:** train = 50,000, test = 10,000
* **Source/reference:** Torchvision auto-download or TFDS — [https://www.tensorflow.org/datasets/catalog/cifar10](https://www.tensorflow.org/datasets/catalog/cifar10)

---

## Level-wise challenge (L1 → L4)

### Level 1 — Baseline

* **Objective:** Transfer-learning baseline (ResNet family).
* **Expected ≥ 85%**
* **Deliverables:** data loader, baseline model, metrics, training curves
* **Notebook:** Level-1 Colab

### Level 2 — Intermediate (augmentation & tuning)

* **Objective:** Improve baseline via augmentation, MixUp, regularization, hyperparameter tuning.
* **Expected ≥ 90%**
* **Reported example:** Stage-1 head training w/ strong aug + MixUp → **~92.5%** val; Stage-2 fine-tune → **95.7%** peak val.
* **Notebook:** Level-2 Colab

### Level 3 — Advanced Architecture

* **Objective:** Custom/attention architectures, interpretability (Grad-CAM).
* **Expected ≥ 91%**
* **Notebook:** Level-3 Colab

### Level 4 — Expert Techniques

* **Objective:** Ensembles, SOTA techniques, or meta-learning.
* **Expected ≥ 93%**
* **Notebook:** Level-4 Colab

(Colab links are provided in the repo `experiments/` and README.)

---

## Quickstart (local / Colab / inference)

### Clone (local)

```bash
git clone https://github.com/Indian24/cifar10-advanced-image-classification.git
cd cifar10-advanced-image-classification
```

### Open in Google Colab

Open any Level notebook in `experiments/` or use the shared Colab links in the README — Colab auto-downloads CIFAR-10.

### Quick inference (Docker)

Build & run example inference API (FastAPI) shipped in `src/deployment/`:

```bash
# build image (example)
docker build -t cifar10-infer .

# run locally on port 8080
docker run --rm -p 8080:8080 cifar10-infer

# health check
curl http://localhost:8080/health

# example predict (JSON)
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"image_base64": "<BASE64_IMAGE>"}'
```

---

## Install & dependencies

Recommended: Python 3.8+ in a venv/conda (reproducible environment).

```bash
# conda example
conda create -n cifar10 python=3.8 -y
conda activate cifar10

# install pinned dependencies
pip install -r requirements.txt

# minimal (quick):
pip install torch torchvision numpy matplotlib seaborn pyyaml fastapi uvicorn onnxruntime
```

`configs/training.yaml` holds defaults for epochs, lr, batch size, two-stage flags, and dataset paths.

---

## How to run (train / evaluate / infer)

### Train (default config)

```bash
python src/training/train.py --config configs/training.yaml
```

### Quick dry run (CI / smoke test)

```bash
python src/training/train.py --config configs/training.yaml --dry_run True
```

### Evaluate

```bash
python src/training/evaluate.py --checkpoint results/checkpoints/best.pth
```

### Inference (script)

```bash
python src/deployment/infer.py --checkpoint results/checkpoints/best.pth --image samples/airplane.png
```

All scripts read `configs/training.yaml` by default. CLI arguments can override config values.

---

## Project structure & key files

```
cifar10-advanced-image-classification/
├── configs/
│   └── training.yaml           # training hyperparams & paths
├── experiments/                # Colab notebooks (Level1..4)
├── src/
│   ├── data/
│   │   └── dataset.py          # dataloaders & augmentations
│   ├── models/
│   │   └── baseline.py         # ResNet wrappers, model utils
│   ├── training/
│   │   ├── train.py            # training loop (two-stage supported)
│   │   └── evaluate.py         # evaluation & metrics
│   └── deployment/
│       ├── infer.py            # CLI inference
│       └── infer_api.py        # FastAPI inference app
├── results/                    # checkpoints, logs, figures
├── requirements.txt
├── README.md
└── .github/
    └── workflows/ci.yml        # CI smoke tests (example)
```

**Key files**

* `configs/training.yaml` — default hyperparameters (epochs, lr, batch_size, two_stage, seed).
* `src/training/train.py` — training loop: supports MixUp, two-stage, checkpointing, logging hooks.
* `src/data/dataset.py` — transformations and dataloaders (auto-download CIFAR-10).
* `src/deployment/infer_api.py` — FastAPI app for serving predictions.
* `experiments/*.ipynb` — guided Colab notebooks for each level.

---

## Examples & expected outputs

**Start training**

```bash
python src/training/train.py --config configs/training.yaml
```

**Sample console excerpt**

```
Stage 1: training head only.
Epoch [1/12], train_loss: 1.9093, train_acc: 0.3448, val_acc: 0.3516, time: 85.0s
...
Epoch [5/12], train_loss: 0.3452, train_acc: 0.9152, val_acc: 0.9250, time: 82.1s
Stage 2: unfreezing all parameters (fine-tune).
...
Final test accuracy: 0.957
```

**Evaluate**

```bash
python src/training/evaluate.py --checkpoint results/checkpoints/best.pth
# prints: overall accuracy, per-class precision/recall, confusion matrix
```

---

## Experiment reproducibility & two-stage summary

**Default `configs/training.yaml` (example)**

```yaml
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
seed: 42
```

**Determinism**

* Set `seed` in config; optionally set:

```python
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

(note: determinism may slow training)

**Two-stage training (recommended reproducible recipe)**

1. **Stage-1 (head training)**

   * Freeze backbone; train only classifier head with strong augmentations + MixUp. Reported val ≈ **92.5%**.
2. **Stage-2 (fine-tune)**

   * Unfreeze backbone; lower LR, lighter augmentation; fine-tune end-to-end. Reported peak val **95.7%**.

**Reproducibility checklist**

```bash
# 1) clone
git clone https://github.com/Indian24/cifar10-advanced-image-classification.git
cd cifar10-advanced-image-classification

# 2) install
conda create -n cifar10 python=3.8 -y
conda activate cifar10
pip install -r requirements.txt

# 3) run training
python src/training/train.py --config configs/training.yaml

# 4) evaluate best checkpoint
python src/training/evaluate.py --checkpoint results/checkpoints/best.pth
```

---

## Results (reported / example)

|                  Level | Target Accuracy | Reported / Example                          |
| ---------------------: | --------------: | ------------------------------------------- |
|     Level 1 (baseline) |           ≥ 85% | Transfer-learning baseline                  |
| Level 2 (intermediate) |           ≥ 90% | Stage-1 ≈ **92.5%**, Stage-2 peak **95.7%** |
|     Level 3 (advanced) |           ≥ 91% | Custom architecture + Grad-CAM analysis     |
|       Level 4 (expert) |           ≥ 93% | Ensemble/meta strategies (research quality) |

> Note: results vary with seed, compute, augmentation strength, and hyperparameter tuning. Always include full logs & config used for each reported run.

---

## Model card & limitations

* **Model:** ResNet-based classifier (configurable); exported to ONNX / TorchScript for inference.
* **Intended use:** Academic / hiring challenge / demo for CIFAR-10 image classification.
* **Limitations:** Trained on CIFAR-10 (low-res 32×32) — not suitable for high-resolution, real world images without retraining. Model may misclassify visually ambiguous classes (cat ↔ dog) — see confusion matrix and Grad-CAM visualizations in `results/`.
* **Bias & fairness:** CIFAR-10 is a balanced academic dataset; real-world domain shifts will degrade performance.
* **Licensing & data:** CIFAR-10 dataset terms apply; code licensed under MIT.

---

## Deployment & MLOps playbook (for interviews / hiring tests)

**Local demo**

* FastAPI app: `src/deployment/infer_api.py`
* Docker image example: `docker build -t cifar10-infer .` and `docker push <registry>/cifar10-infer:tag`

**Cloud options**

* **GCP Cloud Run**: push image to Google Container Registry (GCR) → deploy Cloud Run (serverless).
* **AWS ECS / ECR**: push image to ECR → create ECS Fargate task.
* **Kubernetes**: add `k8s/deployment.yaml` and `service.yaml` (use a HorizontalPodAutoscaler for traffic scaling).

**CI / CD**

* GitHub Actions for lint / unit tests / smoke training run and Docker build. Example skeleton located in `.github/workflows/ci.yml`.

**Observability**

* Expose Prometheus metrics in the FastAPI app (request count, latency, model version).
* Log structured inference metadata to stdout (for ingestion into Cloud Logging / ELK).
* Integrate W&B / MLflow to store experiment artifacts and metrics.

**Security**

* Scan Docker images (Trivy) in CI, use secrets manager for registry credentials, enforce token rotation.

---

## How to contribute

1. Fork → `feature/<name>` branch.
2. Add tests & update `configs/`.
3. Run CI checks locally: `pytest` and `flake8`.
4. Create PR with experiment logs, config, and expected outputs.

**PR checklist**

* [ ] Lint passes (flake8)
* [ ] Unit tests added/updated
* [ ] Configs documented
* [ ] Notebook outputs reproducible
* [ ] Results artifacts in `results/` (when applicable)

---

## Resume bullets & interview talking points

Use these exact lines in your resume/LinkedIn and talk through them in interviews:

* “Built **CIFAR-10 Advanced Image Classification** pipeline with two-stage training and MixUp; achieved peak validation **95.7%** after head-only and end-to-end fine-tuning.”
* “Packaged model as a Dockerized FastAPI service; exported model to ONNX; added CI smoke tests and a deployment playbook for Cloud Run / ECS.”
* “Integrated experiment tracking (W&B/MLflow patterns), Grad-CAM explainability, and per-class analysis for production readiness.”

**Prep answers:**

* Be ready to explain why two-stage training improves stability, how MixUp helps generalization, and tradeoffs of export formats (TorchScript vs ONNX).

---

## Example CI skeleton (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with: python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: flake8 src
      - run: pytest -q
      - run: python src/training/train.py --config configs/training.yaml --dry_run True
```

---

## License & contact

* **License:** MIT (see `LICENSE`)
* **Author / repo owner:** Indian24
* **Contact:** open an issue or PR on the repository for questions / collaboration.

---

## Notes / caveats

* **Auto-download:** CIFAR-10 is auto-downloaded by Torchvision when `download=True`. If behind a firewall, download manually and point `configs/training.yaml` `data_root`.
* **GPU recommended:** CPU training will be slow. Use Colab GPU for experiments.
* **Windows CRLF:** Notebooks and some files may trigger LF/CRLF warnings on Windows — normalize before final commits.
* **Determinism may slow training.**

---

## Appendix — useful commands

```bash
# smoke test (one epoch small batch)
python src/training/train.py --config configs/training.yaml --dry_run True

# run full training
python src/training/train.py --config configs/training.yaml

# evaluate
python src/training/evaluate.py --checkpoint results/checkpoints/best.pth

# build & run inference service
docker build -t cifar10-infer .
docker run --rm -p 8080:8080 cifar10-infer

# git workflow
git status
git add .
git commit -m "experiment: two-stage run"
git push origin main
```

---

