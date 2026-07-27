## Anonymous Review Repository

This repository is provided exclusively for anonymous peer review of the accompanying NeurIPS 2026 Evaluation & Datasets submission. The complete benchmark is shared to facilitate scientific evaluation during the review process. An official public release and licensing terms will follow institutional approval.

# BloodCellBank-Atlas: A Benchmark for Personalized Federated Learning under Extreme Class-Disjoint Label Skew (EC-DLS)

> **Anonymous NeurIPS 2026 Submission (Evaluation & Datasets Track)**

This repository contains the **anonymous implementation** accompanying the NeurIPS 2026 submission:

> **BloodCellBank-Atlas: A Unified Benchmark for Personalized Federated Learning under Extreme Class-Disjoint Label Skew**

The repository includes:

- the reference implementation of **SA-PFL (Semantic Anchor Guided Personalized Federated Learning)**,
- sample benchmark data,
- semantic descriptions,
- taxonomy files,
- benchmark partitions,
- and scripts required to reproduce the reported experiments.

The complete benchmark, full dataset, codebase, and documentation will be publicly released upon acceptance.

---

# Overview

BloodCellBank-Atlas is a unified benchmark for evaluating **Personalized Federated Learning (PFL)** under **Extreme Class-Disjoint Label Skew (EC-DLS)**.

Unlike conventional non-IID federated benchmarks, EC-DLS simulates an extreme but clinically meaningful setting where different institutions observe **mutually exclusive subsets of blood-cell classes**. This benchmark explicitly studies the trade-off between

- **Local Personalization**
- **Global Generalization**

under severe label heterogeneity.

The benchmark harmonizes three blood-cell datasets under a unified taxonomy:

| Dataset | Classes | Cell Types |
|---------|---------:|------------|
| **BLC28** | 28 | RBC + WBC + Platelets |
| **Matek-19** | 15 | WBC |
| **Multi-focus WBC** | 18 | WBC |

Each class is additionally associated with expert-reviewed morphological semantic descriptions used by the proposed SA-PFL framework.

---

# Repository Structure
```text
Neurips2026_submission/

README.md
requirements.txt
environment.yml
LICENSE.txt
setup.sh

stage1.py
stage2.py

metadata/
│
├── semantic_descriptions.csv
├── taxonomy.csv
└── README.md

sample_dataset/
│
├── train/
├── test/
└── README.md

configs/
│
├── blc28/
├── matek19/
└── multifocus/

checkpoints/
│
└── clip_teacher/

output/
│
├── checkpoints/
├── logs/
└── metrics/

docs/
│
├── benchmark.md
├── dataset.md
├── reproducibility.md
└── taxonomy.md
```

---

# SA-PFL Pipeline

The proposed framework consists of **two stages**.

## Stage 1 - Semantic-Anchor Guided Global Learning

Stage 1 performs collaborative federated optimization across participating clients.

The framework combines

- shared MobileNetV3 backbone
- frozen vision-language teacher
- semantic knowledge distillation
- FedProx regularization
- FedAvgM server aggregation

to improve **global generalization** under EC-DLS.

Output:

```text
global_model.pth
global_accuracy.npy
```

---

## Stage 2 - Client Personalization

Each client initializes from the globally trained model.

The classifier head is adapted locally while preserving the globally learned semantic representation.

Stage 2 improves **local personalization** without sacrificing global performance.

Output:

```text
stage2_local_accuracy.npy
```

---

# Dataset

The repository contains a representative subset of BloodCellBank-Atlas for anonymous evaluation.

The complete benchmark will be publicly released upon paper acceptance.

Dataset layout:

```text
sample_dataset/

train/

    class_1/
    class_2/
    ...

test/

    class_1/
    class_2/
    ...
```

Supported formats

- png
- jpg
- jpeg
- tif
- tiff

Images are resized to

```
224 × 224
```

using ImageNet normalization.

---

# Semantic Descriptions

Each blood-cell class is associated with

- expert-reviewed morphological descriptions
- standardized semantic attributes
- unified taxonomy labels

The semantic descriptions are used during Stage 1 for semantic-guided knowledge distillation.

---

# Benchmark Protocol

The benchmark supports two federated settings.

| Protocol | Description |
|----------|-------------|
| EC-DLS | Fully disjoint client label spaces |
| Dirichlet | Conventional non-IID federated learning |

For EC-DLS,

- every class is assigned to exactly one client,
- clients receive mutually exclusive class subsets,
- all experiments are repeated over multiple random partition seeds.

---

# Configuration

Important hyperparameters can be modified directly in the scripts.

| Parameter | Default |
|------------|---------:|
| Communication Rounds | 500 |
| Local Epochs | 5 |
| Batch Size | 32 |
| KD Temperature | 4.0 |
| Maximum KD Weight | 0.8 |
| FedProx μ | 2×10⁻³ |
| FedAvgM Momentum | 0.9 |
| Random Seed | 43 |

---

# Installation

Create the environment

```bash
pip install -r requirements.txt
```

or

```bash
conda env create -f environment.yml
conda activate bloodcellbank
```

---

# Running Stage 1

```bash
python stage1.py
```

Outputs

```
global_model.pth
stage1_global_acc.npy
```
---

# Running Stage 2

```bash
python stage2.py
```

Outputs

```
personalized_model/
stage2_local_accuracy.npy
```

---

# Reproducing Paper Results

To reproduce the reported experiments:

1. Prepare the benchmark dataset.
2. Generate benchmark partitions.
3. Train Stage 1.
4. Run Stage 2 personalization.
5. Evaluate

- Local Personalization
- Global Generalization

under EC-DLS or conventional non-IID settings.

Configuration files corresponding to the experiments reported in the paper are provided in the `configs/` directory.

---

# Reproducibility

Unless otherwise specified, experiments use

- Python 3.10
- PyTorch
- MobileNetV3-Large
- CUDA GPU
- Random Seed = 43

The repository includes

- deterministic partition generation,
- fixed benchmark splits,
- semantic descriptions,
- unified taxonomy,
- experiment configurations.

---

# Citation

This repository accompanies the anonymous NeurIPS 2026 submission.

The complete benchmark, source code, trained models, and documentation will be publicly released upon acceptance.
