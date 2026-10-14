# Soybean Disease Detection using EfficientFormer

## Overview

This project performs multiclass soybean disease classification using EfficientFormer.

Classes:
- Healthy
- Rust
- Mosaic
- Frog Leaf Eye
- Spectoria Brown Spot
- Caterpillar and Semilooper Pest Attack

## Model

EfficientFormer-L1
EfficientFormer-L3

## Dataset

Dataset not included due to size limitations.

Download dataset from:
https://data.mendeley.com/datasets/hkbgh5s3b7/1

(For time being we are only using Image Dataset, not the UAV one)
## Results

Best Validation Accuracy:
78.31%

Best Test Accuracy:
79.5%

## Setup

pip install -r requirements.txt

## Training

python src/train.py

## Evaluation

python src/evaluate.py