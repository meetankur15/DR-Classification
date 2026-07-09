# DR-Classification
# Diabetic Retinopathy Detection using Ordinal Graph Attention-Informed CNN 

## Overview

This repository contains the implementation of a Ordinal Graph Attention-Informed CNN framework for automated diabetic retinopathy (DR) grading from retinal fundus images.The proposed approach combines pre-trained deep learning models for feature extraction.
## Features

- Pre-trained CNN/Transformer feature extraction
- Graph Convolutional Network (GCN) for feature refinement
- Five-class diabetic retinopathy grading
- Confidence estimation
- Uncertainty estimation using Monte Carlo Dropout
- Evaluation on multiple public datasets

---

## Datasets

The experiments were conducted on the following publicly available datasets:

- APTOS 2019 Blindness Detection

Please download the datasets from their official sources and place them in the following directory structure.

```
data/
│── APTOS2019/


## Project Structure

```
.
├── data/
├── preprocessing/
├── feature_extraction/
├── graph_construction/
├── models/
├── checkpoints/
├── utils/
├── train.py
├── test.py
├── inference.py
├── requirements.txt
├── README.md
└── LICENSE
