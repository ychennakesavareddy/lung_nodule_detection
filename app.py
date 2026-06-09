import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os
from datetime import datetime
import cv2
import gradio as gr

# =========================
# CONFIGURATION
# =========================

class Config:
    IMAGE_SIZE = 224
    NUM_CLASSES = 3
    CLASS_NAMES = ['Normal', 'Benign', 'Malignant']

config = Config()

# =========================
# MODEL ARCHITECTURE
# =========================

class LungCancerModel(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCancerModel, self).__init__()

        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        self.features = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# =========================
# IMAGE PROCESSING
# =========================

def preprocess_image(image):

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    return transform(image).unsqueeze(0)

def apply_histogram_equalization(image):

    try:
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        equalized = cv2.equalizeHist(image)

        return Image.fromarray(equalized)

    except:
        return image

def apply_segmentation(image):

    try:
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, segmented = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return Image.fromarray(segmented)

    except:
        return image

def apply_filtering(image):

    try:
        if isinstance(image, Image.Image):
            image = np.array(image)

        filtered = cv2.medianBlur(image, 5)

        return Image.fromarray(filtered)

    except:
        return image

def generate_heatmap(image, model, device):

    try:
        input_tensor = preprocess_image(image).to(device)

        model.eval()

        with torch.no_grad():

            features = model.features(input_tensor)

            weights = torch.mean(
                features,
                dim=(2, 3),
                keepdim=True
            )

            heatmap = torch.sum(features * weights, dim=1)

            heatmap = torch.relu(heatmap)

            heatmap = heatmap.squeeze().cpu().numpy()

            if heatmap.max() > heatmap.min():
                heatmap = (
                    heatmap - heatmap.min()
                ) / (
                    heatmap.max() - heatmap.min()
                )

        if isinstance(image, Image.Image):
            original_image = np.array(image)
        else:
            original_image = image

        if len(original_image.shape) == 3:
            original_image = cv2.cvtColor(
                original_image,
                cv2.COLOR_RGB2GRAY
            )

        heatmap_resized = cv2.resize(
            heatmap,
            (
                original_image.shape[1],
                original_image.shape[0]
            )
        )

        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_JET
        )

        heatmap_colored = cv2.cvtColor(
            heatmap_colored,
            cv2.COLOR_BGR2RGB
        )

        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(
                original_image,
                cv2.COLOR_GRAY2RGB
            )

        superimposed = cv2.addWeighted(
            original_image,
            0.6,
            heatmap_colored,
            0.4,
            0
        )

        return Image.fromarray(superimposed)

    except Exception as e:
        print(f"Heatmap error: {e}")
        return image

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(image):

    try:

        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2GRAY
            )

        hist = cv2.calcHist(
            [image],
            [0],
            None,
            [256],
            [0, 256]
        )

        hist = hist / (hist.sum() + 1e-10)

        entropy = -np.sum(
            hist * np.log2(hist + 1e-10)
        )

        contrast = np.std(image)

        energy = np.sum(hist ** 2)

        mean_intensity = np.mean(image)

        return {
            'entropy': float(entropy),
            'contrast': float(contrast),
            'energy': float(energy),
            'mean_intensity': float(mean_intensity)
        }

    except:
        return {
            'entropy': 0,
            'contrast': 0,
            'energy': 0,
            'mean_intensity': 0
        }

# =========================
# LOAD MODEL
# =========================

def load_model():

    print("\n==================================================")
    print("Initializing Model...")
    print("==================================================")

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Device: {device}")

    model = LungCancerModel(
        num_classes=config.NUM_CLASSES
    )

    model_path = "models/lung_cancer_model.pth"

    try:

        print(f"\nLoading model from: {model_path}")

        checkpoint = torch.load(
            model_path,
            map_location=device
        )

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:

            model.load_state_dict(
                checkpoint['model_state_dict']
            )

        else:
            model.load_state_dict(checkpoint)

        print("\n✅ Model loaded successfully!")

    except Exception as e:

        print(f"\n❌ Model loading error: {e}")

        print("\n⚠️ Using demonstration model")

        for param in model.parameters():
            param.data.normal_(0, 0.02)

    model.to(device)

    model.eval()

    return model, device

# =========================
# PREDICTION
# =========================

def predict_lung_cancer(image, model, device):

    try:

        input_tensor = preprocess_image(image).to(device)

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.nn.functional.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        predicted_class = config.CLASS_NAMES[
            predicted.item()
        ]

        confidence_score = confidence.item()

        equalized = apply_histogram_equalization(image)

        segmented = apply_segmentation(image)

        filtered = apply_filtering(image)

        heatmap = generate_heatmap(
            image,
            model,
            device
        )

        features = extract_features(image)

        return (
            image,
            equalized,
            segmented,
            heatmap,
            filtered,
            f"## Prediction: {predicted_class}",
            f"## Confidence: {confidence_score:.2%}",
            f"""
### Features

- Entropy: {features['entropy']:.4f}
- Contrast: {features['contrast']:.4f}
- Energy: {features['energy']:.4f}
- Mean Intensity: {features['mean_intensity']:.2f}
            """,
            f"### Analysis Time: {datetime.now()}"
        )

    except Exception as e:

        print(f"Prediction Error: {e}")

        return (
            None,
            None,
            None,
            None,
            None,
            "Error",
            "Error",
            "Error",
            "Error"
        )

# =========================
# INTERFACE
# =========================

model, device = load_model()

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Lung Cancer Detection System"
) as demo:

    gr.Markdown("""

# 🫁 Lung Cancer Detection System

### Developed By
**Chenna Kesava Reddy Yenugu**

📍 Kadapa, Andhra Pradesh  
📞 +91 77028 50533  
📧 c.yenugu.tech@gmail.com  

🔗 LinkedIn: https://linkedin.com/in/ychennakesavareddy  
🔗 GitHub: https://github.com/ychennakesavareddy  
🔗 Hugging Face: https://huggingface.co/yenugu  
🔗 Portfolio: https://chennayenugu.ccbp.tech  

---

## 🚀 AI Powered Lung Cancer Detection using Deep Learning

### Features
- ResNet50 CNN Architecture
- CT Scan Analysis
- Heatmap Visualization
- Segmentation
- Feature Extraction
- Risk Assessment

---

## 📥 Direct Model Download

https://huggingface.co/yenugu/lung_cancer_model/resolve/main/lung_cancer_model.pth

---

""")

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Upload CT Scan"
            )

            analyze_btn = gr.Button(
                "🔍 Analyze Image",
                variant="primary"
            )

        with gr.Column():

            prediction_out = gr.Markdown()

            confidence_out = gr.Markdown()

            features_out = gr.Markdown()

            time_out = gr.Markdown()

    with gr.Row():

        original_out = gr.Image(label="Original")

        equalized_out = gr.Image(label="Enhanced")

        segmented_out = gr.Image(label="Segmented")

    with gr.Row():

        heatmap_out = gr.Image(label="Heatmap")

        filtered_out = gr.Image(label="Filtered")

    analyze_btn.click(
        fn=lambda image: predict_lung_cancer(
            image,
            model,
            device
        ),
        inputs=input_image,
        outputs=[
            original_out,
            equalized_out,
            segmented_out,
            heatmap_out,
            filtered_out,
            prediction_out,
            confidence_out,
            features_out,
            time_out
        ]
    )

if __name__ == "__main__":

    print("\n============================================================")
    print("🚀 Starting Lung Cancer Detection System")
    print("============================================================\n")

    demo.launch(server_name="0.0.0.0")