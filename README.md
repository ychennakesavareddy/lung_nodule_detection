# 🫁 Lung Cancer Detection & Classification System

<p align="center">
  <img src="assets/banner.png" width="900">
</p>

<p align="center">

[![IEEE Paper](https://img.shields.io/badge/IEEE-Paper-0066CC)](https://ieeexplore.ieee.org/document/11539587)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.0+-FF6600)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/yenugu)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</p>

---

## 📖 Table of Contents

- Overview
- Features
- Model Architecture
- Dataset
- Installation
- Requirements
- Usage
- Gradio Interface
- Image Processing Pipeline
- Explainable AI (Grad-CAM)
- Training Configuration
- Results
- Project Structure
- Model Download
- Research Paper
- Citation
- License
- Contact

---

# 📋 Overview

Lung cancer remains one of the leading causes of cancer-related deaths worldwide. Early identification of malignant pulmonary nodules can significantly improve patient outcomes.

This project presents a Deep Learning-based Lung Cancer Detection and Classification System built using a transfer-learning approach based on ResNet-50. The system analyzes CT scan images and classifies lung nodules into:

- ✅ Normal
- ⚠️ Benign
- 🔴 Malignant

In addition, the platform provides:

- Explainable AI through Grad-CAM
- Automated image enhancement
- Feature extraction analysis
- Interactive Gradio web interface
- Diagnostic report generation

---

# ⚠️ Medical Disclaimer

This project is intended for:

- Research
- Educational purposes
- AI experimentation

It is NOT a medical device and must NOT be used for clinical diagnosis, treatment decisions, or patient care.

Always consult qualified medical professionals for healthcare decisions.

---

# 🎯 Features

- 🧠 ResNet-50 Transfer Learning
- 🔥 Grad-CAM Explainability
- 🖼️ CT Scan Analysis
- 📊 Feature Extraction Metrics
- 🌐 Gradio Interface
- ⚡ Fast Inference
- 📈 Performance Visualization
- 📄 Diagnostic Summary Generation

---

# 🧠 Model Architecture

```text
Input CT Scan (224×224×3)
        │
        ▼
ResNet-50 Backbone
(ImageNet Pretrained)
        │
        ▼
Feature Extraction
(2048 Features)
        │
        ▼
Adaptive Average Pooling
        │
        ▼
Dropout (0.5)
        │
        ▼
Linear Layer (2048 → 512)
        │
        ▼
ReLU Activation
        │
        ▼
Dropout (0.3)
        │
        ▼
Linear Layer (512 → 3)
        │
        ▼
Softmax Output

Normal | Benign | Malignant
```

## Specifications

| Property | Value |
|-----------|--------|
| Framework | PyTorch |
| Backbone | ResNet-50 |
| Input Size | 224 × 224 |
| Classes | 3 |
| Parameters | ~25M |
| Transfer Learning | Yes |
| Explainability | Grad-CAM |

---

# 📊 Dataset

Dataset Used:

**IQ-OTHNCCD Lung Cancer Dataset**

Directory Structure:

```text
datasets/
└── IQ-OTHNCCD/
    ├── Normal cases/
    ├── Benign cases/
    └── Malignant cases/
```

Class Distribution:

| Class | Samples |
|---------|---------|
| Normal | 200+ |
| Benign | 150+ |
| Malignant | 200+ |

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/ychennakesavareddy/lung_nodule_detection.git

cd lung_nodule_detection
```

## Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv venv

source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.19.0
Pillow>=8.0.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
gradio>=3.0.0
```

---

# 💻 Usage

Run Application

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

---

# 🔬 Image Processing Pipeline

| Step | Method |
|--------|----------|
| Resize | 224×224 |
| Contrast Enhancement | Histogram Equalization |
| Segmentation | OTSU Thresholding |
| Denoising | Median Filter |
| Enhancement | Morphological Dilation |

---

# 🔥 Explainable AI

Grad-CAM is used to visualize regions contributing to model predictions.

### Heatmap Interpretation

🔴 Red → Strong Contribution

🟡 Yellow → Moderate Contribution

🔵 Blue → Low Contribution

Visualization Formula:

```text
60% Original Image
+
40% Grad-CAM Heatmap
=
Final Visualization
```

---

# 📈 Training Configuration

```python
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50

LEARNING_RATE = 0.001

OPTIMIZER = Adam

SCHEDULER = StepLR(
    step_size=10,
    gamma=0.1
)
```

---

# 📊 Results

| Metric | Value |
|----------|--------|
| Accuracy | 96.82% |
| Precision | 96% |
| Recall | 95% |
| F1 Score | 95% |

Per-Class Results:

| Class | Precision | Recall | F1 |
|---------|---------|---------|---------|
| Normal | 0.96 | 0.95 | 0.95 |
| Benign | 0.94 | 0.93 | 0.93 |
| Malignant | 0.97 | 0.98 | 0.97 |

---

# 📂 Project Structure

```text
lung_nodule_detection/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── models/
│   └── lung_cancer_model.pth
│
├── datasets/
│   └── IQ-OTHNCCD/
│
├── evaluation_results/
│
├── assets/
│   ├── banner.png
│   └── demo.gif
│
└── external_validation.py
```

---

# 📥 Download Pre-trained Model

### Hugging Face Repository

https://huggingface.co/yenugu/lung_cancer_model

Download:

```bash
wget https://huggingface.co/yenugu/lung_cancer_model/resolve/main/lung_cancer_model.pth
```

Place Model:

```text
models/
└── lung_cancer_model.pth
```

---

# 📄 Research Paper

Title:

**Deep Learning-Based Lung Nodule Classification with Explainable Feature Analysis**

Conference:

**ICICICT 2026**

IEEE Paper:

https://ieeexplore.ieee.org/document/11539587

---

# 📝 Citation

```bibtex
@INPROCEEDINGS{11539587,
author={Yenugu, Chenna Kesava Reddy},
booktitle={2026 International Conference on Intelligent Computing, IoT, and Communication Technologies (ICICICT)},
title={Deep Learning-Based Lung Nodule Classification with Explainable Feature Analysis},
year={2026},
pages={1-6},
doi={10.1109/ICICICT66501.2026.11539587}
}
```

---

# 🤝 Contributing

Pull Requests and Contributions are welcome.

```bash
git checkout -b feature/new-feature

git commit -m "Add feature"

git push origin feature/new-feature
```

---

# 📄 License

Distributed under the MIT License.

See LICENSE for details.

---

# 📧 Contact

**Yenugu Chenna Kesava Reddy**

GitHub:
https://github.com/ychennakesavareddy

LinkedIn:
https://www.linkedin.com/in/ychennakesavareddy

Hugging Face:
https://huggingface.co/yenugu

Portfolio:
https://chennareddy.in

Email:
c.yenugu.tech@gmail.com

---

# 🙏 Acknowledgements

- IEEE
- IQ-OTHNCCD Dataset Authors
- PyTorch Team
- Gradio Team
- Open Source Community

---

<div align="center">

## ⭐ Star this repository if you found it useful ⭐

Made with ❤️ for Lung Cancer Detection Research

</div>
