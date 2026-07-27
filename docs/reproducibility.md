# Reproducibility Guide

This repository accompanies the anonymous NeurIPS 2026 submission:

> BloodCellBank-Atlas: A Benchmark for Personalized Federated Learning under Extreme Class-Disjoint Label Skew

## Environment

Python 3.10

PyTorch

CUDA GPU (optional)

Transformers

## Random Seed

All reported experiments use

Seed = 43

unless otherwise specified.

---

## Stage 1

Implements Algorithm 1

Semantic-Anchor Guided Global Learning

Outputs

- global_model.pth
- global_accuracy.npy

---

## Stage 2

Implements Algorithm 2

Client-Specific Personalization

Input

global_model.pth

Outputs

personalized models

local_accuracy.npy

---

## Benchmark Protocols

Supported protocols

- EC-DLS
- Dirichlet non-IID

The benchmark partitions are deterministic given the random seed.

---

## Reproducing Results

Run

```bash
python stage1.py
```

followed by

```bash
python stage2.py
```

The reported metrics correspond to

- Local Personalization Accuracy

- Global Generalization Accuracy

as described in the paper.