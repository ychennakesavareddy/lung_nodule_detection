# 🫁 Lung Cancer Detection & Classification System

[![IEEE Paper](https://img.shields.io/badge/IEEE-Paper-0066CC)](https://ieeexplore.ieee.org/document/11539587)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.0+-FF6600)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

# 📋 Overview

This project implements a **Deep Learning-based Lung Cancer Detection and Classification System** using a **ResNet-50** architecture with transfer learning. The model analyzes CT scan images and classifies lung nodules into three categories:

- ✅ Normal
- ⚠️ Benign
- 🔴 Malignant

The system also provides:

- Explainable AI using **Grad-CAM**
- Image enhancement and preprocessing pipeline
- Feature extraction metrics
- Interactive **Gradio Web Interface**
- Cancer stage estimation and diagnostic summary

> **Research Paper:**  
> *Deep Learning-Based Lung Nodule Classification with Explainable Feature Analysis*  
> **Author:** Yenugu Chenna Kesava Reddy  
> **Conference:** ICICICT 2026

---

# 🎯 Key Features

- 🧠 ResNet-50 Transfer Learning Model
- 📊 High Validation Accuracy (96.82%)
- 🔥 Grad-CAM Explainability
- 🖼️ CT Scan Image Analysis
- 📈 Feature Extraction (Entropy, Contrast, Energy)
- 🩺 Cancer Stage Prediction
- 🌐 Gradio-based Web Interface
- ⚡ Fast Inference Pipeline

---

# 🧠 Model Architecture

```text
Input Image (224x224x3)
        │
        ▼
ResNet-50 Backbone
(Pre-trained on ImageNet)
        │
        ▼
Feature Extraction
(2048 Features)
        │
        ▼
AdaptiveAvgPool2D
        │
        ▼
Dropout (0.5)
        │
        ▼
Linear (2048 → 512)
        │
        ▼
ReLU
        │
        ▼
Dropout (0.3)
        │
        ▼
Linear (512 → 3)
        │
        ▼
Output:
Normal | Benign | Malignant
```

## Model Specifications

| Component | Specification |
|------------|--------------|
| Base Model | ResNet-50 |
| Framework | PyTorch |
| Input Size | 224 × 224 |
| Classes | 3 |
| Parameters | ~25 Million |
| Transfer Learning | Yes |
| Explainability | Grad-CAM |

---

# 📊 Dataset

The model is trained on the **IQ-OTHNCCD Lung Cancer Dataset**.

### Dataset Categories

| Class | Samples |
|---------|---------|
| Normal | 200+ |
| Benign | 150+ |
| Malignant | 200+ |

### Dataset Structure

```text
datasets/
└── IQ-OTHNCCD/
    ├── Normal cases/
    │   ├── Normal-1.jpg
    │   ├── Normal-2.jpg
    │   └── ...
    │
    ├── Bengin cases/
    │   ├── Benign-1.jpg
    │   └── ...
    │
    └── Malignant cases/
        ├── Malignant-1.jpg
        └── ...
```

---

# 🚀 Installation

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (Recommended)
- 8GB+ RAM

## Clone Repository

```bash
git clone https://github.com/ychennakesavareddy/lung_nodule_detection.git

cd lung_nodule_detection
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

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

## Run Web Application

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

---

## Train Model

```bash
python train.py --train
```

---

## Command Line Inference

```python
from app import predict_lung_cancer, load_model
from PIL import Image

model, device = load_model()

image = Image.open("sample.jpg")

result = predict_lung_cancer(
    image,
    model,
    device
)

print("Prediction:", result["prediction"])
print("Confidence:", result["confidence"])
print("Stage:", result["stage"])
```

---

# 📱 Gradio Interface

## Input Panel

- CT Scan Upload
- Process Button
- Clear Button
- Exit Button

## Output Panel

- Classification Result
- Confidence Score
- Cancer Stage
- Cure Probability
- Feature Analysis

## Visualization Tabs

### Image Processing

- Original Image
- Histogram Equalization
- OTSU Segmentation
- Median Filtering
- Morphological Dilation

### Nodule Detection

- Grad-CAM Heatmap
- Diagnostic Summary

---

# 🔬 Image Processing Pipeline

| Step | Technique | Purpose |
|--------|----------|----------|
| 1 | Resize (224×224) | Standardize Input |
| 2 | Histogram Equalization | Enhance Contrast |
| 3 | OTSU Thresholding | Lung Segmentation |
| 4 | Median Filtering | Noise Removal |
| 5 | Morphological Dilation | Feature Enhancement |

---

# 🎨 Grad-CAM Visualization

The system provides explainable predictions using **Grad-CAM**.

### Color Interpretation

🔴 Red Regions → High Influence

🟡 Yellow Regions → Medium Influence

🔵 Blue Regions → Low Influence

### Overlay Formula

```text
60% Original Image
+
40% Heatmap
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

NUM_CLASSES = 3

OPTIMIZER = Adam

SCHEDULER = StepLR(
    step_size=10,
    gamma=0.1
)
```

---

# 📊 Performance Metrics

| Class | Precision | Recall | F1 Score |
|---------|----------|--------|----------|
| Normal | 0.96 | 0.95 | 0.95 |
| Benign | 0.94 | 0.93 | 0.93 |
| Malignant | 0.97 | 0.98 | 0.97 |
| Average | 0.96 | 0.95 | 0.95 |

### Overall Results

- Accuracy: **96.82%**
- Train Split: **80%**
- Validation Split: **20%**

---

# 🩺 Cancer Stage Estimation

> **Note:** This stage estimation is for educational/research purposes and should not be used for clinical diagnosis.

| Confidence Range | Stage | Estimated Cure Probability |
|------------------|--------|----------------------------|
| < 30% | Stage I | 85% |
| 30% – 60% | Stage II | 60% |
| 60% – 80% | Stage III | 30% |
| > 80% | Stage IV | 10% |

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
│       ├── Normal cases/
│       ├── Bengin cases/
│       └── Malignant cases/
│
├── evaluation_results/
│
└── external_validation.py
```

---

# 📝 Citation

If you use this project in your research, please cite:

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

Contributions are welcome.

```bash
# Fork Repository

# Create Feature Branch
git checkout -b feature/NewFeature

# Commit Changes
git commit -m "Add New Feature"

# Push Changes
git push origin feature/NewFeature

# Create Pull Request
```

---

# 📄 License

Distributed under the MIT License.

See the LICENSE file for more information.

---

# 📧 Contact

**Yenugu Chenna Kesava Reddy**

- GitHub: https://github.com/ychennakesavareddy
- LinkedIn: https://www.linkedin.com/in/ychennakesavareddy
- Hugging Face: https://huggingface.co/yenugu
- Portfolio: https://chennareddy.in
- Email: c.yenugu.tech@gmail.com

---

# 🙏 Acknowledgments

- IEEE
- IQ-OTHNCCD Dataset Authors
- PyTorch Team
- Gradio Team
- Open Source Community

---

<div align="center">

## ⭐ If you found this project useful, please give it a star ⭐

### Made with ❤️ for Lung Cancer Detection and Early Diagnosis

</div>
