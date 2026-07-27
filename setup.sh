#!/bin/bash

echo "======================================="
echo "BloodCellBank-Atlas Benchmark Setup"
echo "======================================="

python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p output
mkdir -p output/checkpoints
mkdir -p output/metrics
mkdir -p output/logs

echo ""
echo "Installation complete."
echo ""
echo "Example:"
echo "python stage1.py"
echo "python stage2.py"