# 🫁 Lung Cancer Detection & Classification System

[![IEEE Paper](https://img.shields.io/badge/IEEE-Paper-0066CC)](https://ieeexplore.ieee.org/document/11539587)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-3.0+-FF6600)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📋 Overview

This project implements a **deep learning-based lung cancer detection system** using a **ResNet-50** architecture with transfer learning. It achieves **96.82% validation accuracy** on CT scan images and provides explainable predictions through **Grad-CAM visualization**. The system classifies lung nodules into three categories: **Normal**, **Benign**, and **Malignant**.

> **Based on IEEE Paper:** "Deep Learning—Based Lung Nodule Classification with Explainable Feature Analysis" (DOI: 10.1109/ICICICT66501.2026.11539587)

## 🎯 Key Features

- **High Accuracy**: 96.82% classification accuracy on validation dataset
- **ResNet-50 Architecture**: Transfer learning with pre-trained ImageNet weights
- **Grad-CAM Visualization**: Heatmap overlay showing regions influencing predictions
- **Multi-stage Analysis**:
  - Histogram Equalization for contrast enhancement
  - OTSU Thresholding for segmentation
  - Median Filtering for noise reduction
  - Morphological Dilation for feature enhancement
- **Feature Extraction**: Entropy, contrast, and energy calculations
- **Cancer Stage Prediction**: Stage I-IV classification with cure probability estimates
- **Interactive Web Interface**: Built with Gradio for easy clinical use

## 🧠 Model Architecture
