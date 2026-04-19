---

title: Lung Cancer Detection
emoji: 🫁
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.29.0"
python_version: "3.10"
app_file: app.py
pinned: false
-------------

# 🫁 Lung Cancer Detection System

A deep learning–based research system for detecting and classifying lung cancer from CT scan images using a ResNet50-based convolutional neural network.
The model achieves **96.82% validation accuracy** on the IQ-OTH/NCCD lung CT dataset and includes visual explainability using heatmaps.

> ⚠️ **For research and educational purposes only. Not for clinical use.**

---

## 🎯 Key Features

* **High Accuracy**: 96.82% validation accuracy
* **Multi-Class Classification**:

  * Normal
  * Benign
  * Malignant
* **Explainable AI**:

  * Heatmap-based nodule localization
* **Cancer Stage Estimation**:

  * Stage I–IV (confidence-based)
* **Risk Assessment**:

  * Confidence-driven reliability indicators
* **Interactive Web Interface**:

  * Built using Gradio
  * Deployed on Hugging Face Spaces

---

## 🚀 How It Works

1. Upload a lung CT scan image
2. The image is preprocessed and normalized
3. A ResNet50-based CNN extracts deep features
4. The classifier predicts the cancer category
5. A heatmap highlights regions influencing the decision
6. Risk level and stage estimation are generated

---

## 📊 Model Performance

| Class     | Precision | Recall  | F1-Score | Support |
| --------- | --------- | ------- | -------- | ------- |
| Normal    | 93.18%    | 98.80%  | 95.91%   | 83      |
| Benign    | 94.74%    | 75.00%  | 83.72%   | 24      |
| Malignant | 100.00%   | 100.00% | 100.00%  | 113     |

**Overall Accuracy:** **96.82%**

> Evaluation performed on a held-out test set to assess generalization.

---

## 🏥 Dataset

**IQ-OTH/NCCD Lung Cancer Dataset**

* **Total Images:** 1,190 CT scan slices
* **Classes:**

  * Normal: 55 cases
  * Benign: 15 cases
  * Malignant: 40 cases

**Source:**
Iraq-Oncology Teaching Hospital /
National Center for Cancer Diseases (NCCD)

> Dataset used strictly for academic research.

---

## 🔬 Technical Architecture

* **Backbone:** ResNet50 (ImageNet pre-trained)
* **Classifier:** Fully connected layers with dropout
* **Framework:** PyTorch
* **Input Resolution:** 224 × 224
* **Optimization:** Adam optimizer
* **Training Epochs:** 50
* **Explainability:** Feature-based heatmap visualization

---

## 🌐 Live Demo

👉 https://huggingface.co/spaces/yenugu/lung_cancer_detection

---

## 🛠️ Local Development

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/lung-nodule-detection.git
cd lung-nodule-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
python app.py
```

---

## 📌 Notes

* Model is loaded dynamically from Hugging Face Model Hub
* Ensure internet connection for first-time model download
* Designed for educational and research purposes only

---

## 👨‍💻 Author

Developed by **Yenugu Chenna Kesava Reddy**

---

## ⭐ Future Improvements

* Integration with real-time hospital systems
* Advanced 3D CT scan analysis
* Improved explainability with Grad-CAM++
* Mobile-friendly interface

---
