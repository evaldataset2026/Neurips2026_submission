# Federated Learning with Personalized Heads for Blood Cell Image Classification

> **Note:** This file contains the demo code for SA-PFL method as part of the review process for NeurIPS2026 submission under the Evaluation & Dataset Benchmark Track. Please note that the full source code will be made publicly available upon paper acceptance. 

---

## Overview

This repository contains the implementation of a **EC-DLS federated learning** framework for image classification using a split-model architecture — a shared global backbone aggregated via FedAvg, paired with per-client personalized classification heads. The system is designed to handle non-IID data distributions across clients, supporting both Dirichlet-based heterogeneity and label-skew partitioning strategies.

---

## Method

The framework adopts a two-component model design:

- **Global Backbone** — A pretrained MobileNetV3-Large feature extractor, aggregated across all clients each round using weighted FedAvg.
- **Personalized Head** — A lightweight linear classifier trained locally per client and never aggregated, allowing each client to specialize for its local class distribution.

At each communication round:
1. Each client receives the latest global backbone weights.
2. Local training is performed for a fixed number of epochs.
3. Updated backbones are sent to the server and averaged via FedAvg.
4. Personalized heads remain on the client side throughout training.

---

## Dataset

The experiments use a custom 28-class image dataset (`blc28`) split into training and test sets. Images are resized to 224×224 and normalized using ImageNet statistics.

```
data/
├── blc28_train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── blc28_test/
    ├── class_0/
    ├── class_1/
    └── ...
```

Supported image formats: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`

---

## Data Partitioning

Two non-IID partitioning strategies are supported:

| Strategy | Description |
|---|---|
| `label_skew` | Fully disjoint class assignment — each client holds exclusively a subset of classes |
| `dirichlet` | Dirichlet distribution–based split with concentration parameter `α` controlling heterogeneity |

---

## Configuration

Key hyperparameters are defined at the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `NUM_CLIENTS` | 28 | Number of federated clients |
| `NUM_CLASSES` | 28 | Total number of output classes |
| `ROUNDS` | 500 | Number of communication rounds |
| `LOCAL_EPOCHS` | 5 | Local training epochs per round |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `DIRICHLET_ALPHA` | 0.5 | Dirichlet concentration (used if `dirichlet` split) |
| `CLIENT_SPLIT` | `label_skew` | Partitioning strategy (`label_skew` or `dirichlet`) |
| `SEED` | 43 | Random seed for reproducibility |

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- pandas
- Pillow

Install dependencies:

```bash
pip install torch torchvision numpy pandas pillow
```

---

## Usage

```bash
python filename.py
```

Accuracy logs are saved after every round:

- `client_local_acc_labelskew_nonovelty7.npy` — Per-client local test accuracy across rounds `(NUM_CLIENTS × ROUNDS)`
- `global_acc_labelskew_nonovelty7.npy` — Global test accuracy across rounds `(ROUNDS,)`

---

## Results Logging

Training produces two `.npy` files that can be loaded for analysis:

```python
import numpy as np

client_acc = np.load("client_local_acc_labelskew_nonovelty7.npy")  # shape: (28, 500)
global_acc = np.load("global_acc_labelskew_nonovelty7.npy")         # shape: (500,)
```


---
