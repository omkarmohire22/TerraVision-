# 🛰️ TerraVision: Off-Road Semantic Scene Intelligence

![TerraVision Hero](assets/hero.png)

## 🌟 Overview

**TerraVision** is a state-of-the-art semantic segmentation suite specifically engineered for off-road autonomous navigation in challenging desert environments. Developed for the **Duality AI Offroad Semantic Scene Segmentation Challenge**, it leverages advanced deep learning architectures to provide pixel-precise scene understanding.

By fusing synthetic data generation from Duality AI's Falcon platform with powerful backbones like **DINOv2** and **DeepLabV3+**, TerraVision achieves robust generalization across diverse off-road terrain.

---

## 🚀 Key Features

*   **Dual-Model Intelligence**: 
    *   **DINOv2**: Self-supervised vision transformer backbone for rich feature extraction.
    *   **DeepLabV3+**: ResNet50-based architecture for high-resolution refinement.
*   **Modular Architecture**: Clean, package-based structure (`src/terravision`) for high maintainability.
*   **Real-time Inference API**: High-performance FastAPI backend with a modern interactive frontend.
*   **Professional Evaluation**: Automated metrics (mIoU, Dice, Pixel Acc) and automated failure case analysis.
*   **Mixed-Precision Training**: Optimized for speed and low VRAM usage.

---

## 🛠️ Project Structure

```bash
TerraVision/
├── assets/             # Project brand assets and documentation images
├── configs/            # Configuration files (YAML) for training and models
├── dataset/            # Training and validation data storage
├── frontend/           # Interactive Web UI (HTML/CSS/JS)
├── outputs/            # Model checkpoints, graphs, and predictions
├── scripts/            # Executable scripts for training and testing
│   ├── train.py        # Main DeepLabV3+ training script
│   ├── train_dinov2.py # DINOv2-based training script
│   ├── test.py         # Batch inference on test images
│   └── inference.py    # Single-image diagnostic inference
├── src/
│   └── terravision/    # Core package source code
│       ├── data/       # Data loaders and advanced augmentations
│       ├── models/     # Model architecture definitions
│       ├── training/   # Training loops and Trainer classes
│       ├── evaluation/ # Metrics and evaluation logic
│       └── visualization/ # Plotting and result rendering
└── app.py              # FastAPI Web Application entry point
```

---

## 📊 Dataset & Semantic Classes

The system is trained to recognize 10 critical off-road categories:

| ID | Class | Description |
|---|---|---|
| 100 | **Trees** | High-level vertical obstacles |
| 200 | **Lush Bushes** | Denser vegetation |
| 300 | **Dry Grass** | Low-level terrain coverage |
| 500 | **Dry Bushes** | Arid environment obstacles |
| 550 | **Ground Clutter** | Small rocks and debris |
| 600 | **Flowers** | Dynamic environmental elements |
| 700 | **Logs** | Significant horizontal obstacles |
| 800 | **Rocks** | Navigation hazards |
| 7100 | **Landscape** | Background and ground plane |
| 10000 | **Sky** | Upper horizon |

---

## ⚙️ Installation & Setup

### 1. Requirements
Ensure you have Python 3.9+ and CUDA installed.

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Place the challenge dataset inside the `dataset/` directory. The structure should follow the standard hierarchy required by the loaders in `src/terravision/data/`.

---

## 🪄 Usage

### 🚂 Training the Model
To start the primary training pipeline (DeepLabV3+):
```bash
python scripts/train.py
```

To train the vision transformer (DINOv2) variant:
```bash
python scripts/train_dinov2.py
```

### 🧪 Evaluation & Testing
Generate predictions for the official test set:
```bash
python scripts/test.py
```

Analyze failure cases (saves samples with lowest IoU):
```bash
python scripts/failure_cases.py
```

### 🌐 Launching the Web Application
Start the interactive segmentation server:
```bash
python app.py
```
*Access the UI at: `http://localhost:8000`*

---

## 📈 Performance & Results

TerraVision tracks training progress via TensorBoard and automated graph generation.
- **mIoU**: Primary performance metric.
- **Visual Validation**: Real-time segmentation overlays generated in `outputs/`.

---

## 🗺️ Roadmap
- [ ] Integration of Focal Loss for better handling of class imbalance.
- [ ] Support for ConvNeXt backbones.
- [ ] Quantization for edge device deployment (NVIDIA Orin/Xavier).

---

© 2026 TerraVision | Built for Duality AI Challenge
