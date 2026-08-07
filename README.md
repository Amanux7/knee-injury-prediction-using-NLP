# RSNA Knee Abnormality Detection & NLP Report Extraction

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DINOv2](https://img.shields.io/badge/Meta_AI-DINOv2-1877F2?logo=meta&logoColor=white)](https://github.com/facebookresearch/dinov2)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)

Production-grade, modular 2.5D Computer Vision & Clinical NLP deep learning framework built for the **RSNA Knee Abnormality Detection** competition. 

The pipeline combines **Meta's DINOv2 Vision Transformers**, **3D DICOM MRI Volumetric Resampling**, **Group-Aware Multilabel Stratified Cross-Validation**, and **Negation-Aware Multilingual NLP Extraction** from free-text radiology reports.

---

## 📌 Key Architectural Features

- **2.5D Volumetric MRI Loader (`src/dataset.py`)**:
  - Direct 3D DICOM slice loading via `pydicom`.
  - Smart Z-axis slice ordering using `InstanceNumber` $\rightarrow$ `SliceLocation` $\rightarrow$ `ImagePositionPatient[2]`.
  - Uniform Z-axis resampling & interpolation to exactly 32 slices per 3D scan volume.
  - 3D Min-Max volume intensity normalisation $\frac{\text{vol} - \min}{\max - \min + 1\text{e-}6}$.
  - Offline-safe synthetic fallback for seamless local/CPU development.

- **DINOv2 & 2.5D Sequence Backbone (`src/models.py`)**:
  - Supports Meta's `dinov2_vits14` (384-dim) and `dinov2_vitb14` (768-dim) Vision Transformers.
  - Offline weight loading from local directory caches (e.g., `/content/models_cache/`), `.tar.gz` archives, or `torch.hub`.
  - **Temporal Sequence Aggregation**: Multi-Head Self-Attention (`TemporalMultiHeadAttention`) or 1D Temporal Convolution (`TemporalConv1DBlock`) across 32 slice embeddings.
  - Input tensor shape flexibility: accepts 5D `[B, 32, 3, 224, 224]` and 4D `[B, 32, 224, 224]` scans.

- **Leak-Free Cross-Validation (`src/cross_validation.py`)**:
  - 5-Fold Group-Aware Multilabel Stratified Split via `iterative-stratification`.
  - Group deduplication by `PatientID` / `StudyInstanceUID` guarantees zero data leakage across train and validation splits.

- **Multilingual Clinical NLP Report Extractor (`src/report_labeler.py`)**:
  - Extracts binary/probabilistic labels across all 12 target abnormalities from free-text radiology reports.
  - **Dual Backend**: Zero-shot LLM classification (`transformers` pipeline) and deterministic regex/keyword extractor.
  - **Bidirectional Negation Detection**: Preceding cues (*"no evidence of"*) and trailing cues (*"ACL is intact"*).
  - Supports English, French, Spanish, German, and Portuguese clinical terminology.

- **Competition Macro ROC-AUC Metric (`src/metrics.py`)**:
  - Macro-averaged ROC-AUC across all 12 abnormality classes.
  - Edge-case protection: safely defaults single-class validation folds to 0.5 AUC without runtime failures.

---

## 🎯 Target Abnormality Classes (12 Targets)

| Target | Description | Target | Description |
|---|---|---|---|
| `ACL` | Anterior Cruciate Ligament Tear | `PF OA` | Patellofemoral Osteoarthritis |
| `MCL` | Medial Collateral Ligament Tear | `Effusion` | Joint Fluid Accumulation |
| `Medial Meniscus` | Medial Meniscal Tear | `Synovitis` | Synovial Inflammation |
| `Lateral Meniscus` | Lateral Meniscal Tear | `Baker's` | Popliteal / Baker's Cyst |
| `Medial OA` | Medial Compartment OA | `Contusion` | Bone Marrow Edema / Contusion |
| `Lateral OA` | Lateral Compartment OA | `Fracture` | Tibial / Femoral Fracture |

---

## 📁 Repository Structure

```
knee-injury-prediction-using-NLP/
├── configs/
│   └── baseline_config.yaml    # Centralized hyperparameters & paths
├── src/
│   ├── __init__.py
│   ├── cross_validation.py     # Group-aware multilabel stratified K-Fold
│   ├── dataset.py              # Volumetric 3D DICOM loader & preprocessor
│   ├── metrics.py              # Macro-averaged ROC-AUC competition metric
│   ├── models.py               # DINOv2 2.5D vision backbone & sequence heads
│   └── report_labeler.py       # Clinical NLP report parser with negation handling
├── checkpoints/                # Model weight checkpoints (best_model_fold_X.pth)
├── submission_zero.py          # Offline zero-shot baseline submission generator
├── train.py                    # Main PyTorch training orchestrator with AMP & Cosine Annealing
├── train_folds.csv             # Generated fold split assignments
├── submission.csv              # Formatted Kaggle submission file
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/Amanux7/knee-injury-prediction-using-NLP.git
cd knee-injury-prediction-using-NLP
pip install torch torchvision timm pydicom scikit-learn pandas numpy pyyaml iterative-stratification
```

### 2. Cross-Validation Split Generation

Generate leak-free 5-fold cross-validation splits:

```bash
python -m src.cross_validation --csv data/train.csv --n_folds 5 --seed 42
```

### 3. Model Training & Verification

#### Synthetic Smoke-Test (End-to-End Pipeline Check)
Run 1 mini-epoch on synthetic data to verify GPU/CPU execution:

```bash
python train.py --smoke-test
```

#### Train Specific Fold
Train Fold 0 using the configuration in `configs/baseline_config.yaml`:

```bash
python train.py --fold 0 --config configs/baseline_config.yaml
```

---

## 🧪 Module Verification & Standalone CLI Utilities

Each module includes a self-contained execution runner (`if __name__ == "__main__":`):

```bash
# Verify DICOM Dataset Loader & 3D Min-Max Volume Normalisation
python -m src.dataset

# Run DINOv2 2.5D Model Architecture Verification (5D & 4D Tensors)
python -m src.models

# Test Negation-Aware NLP Report Labeler (EN, FR, ES Test Cases)
python -m src.report_labeler

# Generate Baseline Zero-Shot Submission CSV
python submission_zero.py
```

---

## ⚙️ Configuration (`configs/baseline_config.yaml`)

```yaml
competition:
  name: "rsna-knee-abnormality-detection"
  num_classes: 12
  target_columns:
    - "ACL"
    - "MCL"
    - "Medial Meniscus"
    - "Lateral Meniscus"
    - "Medial OA"
    - "Lateral OA"
    - "PF OA"
    - "Effusion"
    - "Synovitis"
    - "Baker's"
    - "Contusion"
    - "Fracture"

data:
  image_size: [224, 224]
  num_slices: 32
  batch_size: 8
  num_workers: 4

model:
  backbone: "dinov2_vitb14"

training:
  learning_rate: 3.0e-4
  weight_decay: 1.0e-2
  epochs: 10
  seed: 42
```

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE).