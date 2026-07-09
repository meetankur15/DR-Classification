# DR-Classification
# Diabetic Retinopathy Detection using Ordinal Graph Attention-Informed CNN 

## Overview

This repository contains the implementation of a Ordinal Graph Attention-Informed CNN framework for automated diabetic retinopathy (DR) grading from retinal fundus images.The proposed approach combines pre-trained deep learning models for feature extraction.
## Features

- Pre-trained CNN feature extraction
- Five-class diabetic retinopathy grading
- A hybrid CNN–Graph Attention Network architecture for Diabetic Retinopathy grading.
- CORAL-based ordinal regression for Severity-Aware Prediction.
- Dual Attention Mechanism (Grad-CAM and GAT) for Explainability
- Evaluation on public datasets

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
├── RGCNN.ipynb
├── CNN_GAT Explainabilty.ipynb
├── requirements.txt
├── README.md
└── LICENSE
