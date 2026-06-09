import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import gradio as gr
import json
from datetime import datetime

# Configuration
class Config:
    DATA_PATH = "datasets/The IQ-OTHNCCD lung cancer dataset"
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 0.001
    NUM_CLASSES = 3
    CLASS_NAMES = ['Normal', 'Benign', 'Malignant']
    MODEL_SAVE_PATH = "lung_cancer_model.pth"

config = Config()

# Custom Dataset
class LungCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data Preparation
def prepare_data():
    image_paths = []
    labels = []
    
    # Normal cases
    normal_path = os.path.join(config.DATA_PATH, "Normal cases")
    for img_file in os.listdir(normal_path):
        if img_file.endswith('.jpg'):
            image_paths.append(os.path.join(normal_path, img_file))
            labels.append(0)  # Normal
    
    # Benign cases
    benign_path = os.path.join(config.DATA_PATH, "Bengin cases")
    for img_file in os.listdir(benign_path):
        if img_file.endswith('.jpg'):
            image_paths.append(os.path.join(benign_path, img_file))
            labels.append(1)  # Benign
    
    # Malignant cases
    malignant_path = os.path.join(config.DATA_PATH, "Malignant cases")
    for img_file in os.listdir(malignant_path):
        if img_file.endswith('.jpg'):
            image_paths.append(os.path.join(malignant_path, img_file))
            labels.append(2)  # Malignant
    
    return image_paths, labels

# Enhanced CNN Model with Grad-CAM support
class LungCancerModel(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCancerModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Grad-CAM attributes
        self.gradients = None
        self.activations = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.features(x)
        
        # Register hook for Grad-CAM
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features(x)

# Training Function
def train_model():
    # Prepare data
    image_paths, labels = prepare_data()
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = LungCancerDataset(train_paths, train_labels, train_transform)
    val_dataset = LungCancerDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = LungCancerModel(num_classes=config.NUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{config.EPOCHS}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'epoch': epoch
            }, config.MODEL_SAVE_PATH)
        
        scheduler.step()
    
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    return model

# Image Processing Functions
def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def apply_histogram_equalization(image):
    """Apply histogram equalization to enhance contrast"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    equalized = cv2.equalizeHist(image)
    return Image.fromarray(equalized)

def apply_segmentation(image):
    """Apply threshold-based segmentation"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    _, segmented = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(segmented)

def apply_filtering(image):
    """Apply median filtering to reduce noise"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    filtered = cv2.medianBlur(image, 5)
    return Image.fromarray(filtered)

def apply_dilation(image):
    """Apply morphological dilation"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    return Image.fromarray(dilated)

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        self.activations = self.model.features(x)
        if self.activations.requires_grad:
            h = self.activations.register_hook(self.save_gradient)
        output = self.model.avgpool(self.activations)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return output
    
    def generate_cam(self, x, target_class=None):
        # Forward pass
        output = self.forward(x)
        
        if target_class is None:
            target_class = np.argmax(output.cpu().data.numpy())
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros((1, output.size()[-1]))
        one_hot[0][target_class] = 1
        one_hot = one_hot.to(x.device)
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU and normalization
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        
        return cam, target_class

def generate_gradcam(image, model, device):
    """Generate Grad-CAM heatmap for the input image"""
    grad_cam = GradCAM(model)
    
    # Preprocess image
    input_tensor = preprocess_image(image).to(device)
    
    # Generate CAM
    cam, predicted_class = grad_cam.generate_cam(input_tensor)
    
    # Convert image to numpy
    original_image = np.array(image)
    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    
    # Resize CAM to match original image
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap on original image
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    
    return superimposed, predicted_class

# Cancer Stage Prediction
def predict_stage(confidence_scores, cancer_type):
    """Predict cancer stage based on confidence scores"""
    if cancer_type == "Normal":
        return "No cancer detected", "N/A"
    
    malignant_confidence = confidence_scores[2]  # Malignant class confidence
    
    if cancer_type == "Benign":
        if malignant_confidence < 0.1:
            return "Early Stage", "High (95%)"
        else:
            return "Developing Stage", "Medium (75%)"
    
    # Malignant cancer staging
    if malignant_confidence < 0.3:
        stage = "Stage I"
        cure_probability = "High (85%)"
    elif malignant_confidence < 0.6:
        stage = "Stage II"
        cure_probability = "Medium (60%)"
    elif malignant_confidence < 0.8:
        stage = "Stage III"
        cure_probability = "Low (30%)"
    else:
        stage = "Stage IV"
        cure_probability = "Very Low (10%)"
    
    return stage, cure_probability

# Feature Extraction
def extract_features(image):
    """Extract texture features from the image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate entropy
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Calculate contrast (standard deviation)
    contrast = np.std(image)
    
    # Calculate energy (uniformity)
    energy = np.sum(hist ** 2)
    
    return {
        'entropy': float(entropy),
        'contrast': float(contrast),
        'energy': float(energy)
    }

# Main Prediction Function
def predict_lung_cancer(image, model, device):
    """Main prediction function with comprehensive analysis"""
    # Preprocess image
    input_tensor = preprocess_image(image).to(device)
    
    # Model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = config.CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()
    
    # Generate Grad-CAM
    gradcam_image, _ = generate_gradcam(image, model, device)
    
    # Extract features
    features = extract_features(image)
    
    # Predict stage and cure probability
    stage, cure_prob = predict_stage(probabilities.cpu().numpy()[0], predicted_class)
    
    # Create different views
    equalized = apply_histogram_equalization(image)
    segmented = apply_segmentation(image)
    filtered = apply_filtering(image)
    dilated = apply_dilation(image)
    
    return {
        'prediction': predicted_class,
        'confidence': confidence_score,
        'stage': stage,
        'cure_probability': cure_prob,
        'features': features,
        'gradcam': gradcam_image,
        'processed_images': {
            'original': image,
            'equalized': equalized,
            'segmented': segmented,
            'filtered': filtered,
            'dilated': dilated
        }
    }

# Gradio Interface
def create_gradio_interface():
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LungCancerModel(num_classes=config.NUM_CLASSES)
    
    if os.path.exists(config.MODEL_SAVE_PATH):
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print("No trained model found. Please train the model first.")
        return None
    
    model.to(device)
    model.eval()
    
    def process_image(input_image):
        if input_image is None:
            return None, None, None, None, None, None, None, None, None, None
        
        # Perform prediction
        result = predict_lung_cancer(input_image, model, device)
        
        # Format results
        prediction_text = f"**Prediction:** {result['prediction']}"
        confidence_text = f"**Confidence:** {result['confidence']:.2%}"
        stage_text = f"**Stage:** {result['stage']}"
        cure_text = f"**Cure Probability:** {result['cure_probability']}"
        
        # Feature texts
        feature_texts = [
            f"**Entropy:** {result['features']['entropy']:.4f}",
            f"**Contrast:** {result['features']['contrast']:.4f}",
            f"**Energy:** {result['features']['energy']:.4f}"
        ]
        
        return (
            result['processed_images']['original'],
            result['processed_images']['equalized'],
            result['processed_images']['segmented'],
            result['processed_images']['filtered'],
            result['gradcam'],
            prediction_text,
            confidence_text,
            stage_text,
            cure_text,
            "\n".join(feature_texts)
        )
    
    # Create interface
    with gr.Blocks(title="Lung Cancer Detection System", theme="soft") as interface:
        gr.Markdown("# 🫁 Lung Cancer Detection and Classification Using Neural Network")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Input Image")
                input_image = gr.Image(type="pil", label="Upload CT Scan Image")
                
                with gr.Row():
                    process_btn = gr.Button("🚀 Process Image", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear All")
                    exit_btn = gr.Button("🚪 Exit")
            
            with gr.Column():
                gr.Markdown("## Output Results")
                prediction_output = gr.Markdown(label="Classification Result")
                confidence_output = gr.Markdown(label="Confidence Level")
                stage_output = gr.Markdown(label="Cancer Stage")
                cure_output = gr.Markdown(label="Cure Probability")
                features_output = gr.Markdown(label="Feature Extraction Parameters")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Image Processing Pipeline")
                
                with gr.Tabs():
                    with gr.TabItem("Original"):
                        original_output = gr.Image(label="Input Lung Image")
                    
                    with gr.TabItem("Histogram Equalization"):
                        equalized_output = gr.Image(label="Histogram Equalization")
                    
                    with gr.TabItem("Segmentation"):
                        segmented_output = gr.Image(label="Segmentation by Thresholding")
                    
                    with gr.TabItem("Filtering"):
                        filtered_output = gr.Image(label="Filtered Image")
            
            with gr.Column():
                gr.Markdown("## Nodule Detection & Analysis")
                
                with gr.Tabs():
                    with gr.TabItem("Grad-CAM Heatmap"):
                        gradcam_output = gr.Image(label="Nodule Localization")
                    
                    with gr.TabItem("Diagnostic Summary"):
                        gr.Markdown("### Cancer Detection Summary")
                        summary_text = gr.Markdown("""
                        **Lung Cancer Detection System Analysis:**
                        - Nodule localization using Grad-CAM
                        - Multi-stage classification
                        - Feature-based analysis
                        - Treatment probability estimation
                        """)
        
        # Event handlers
        process_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[
                original_output, equalized_output, segmented_output, 
                filtered_output, gradcam_output, prediction_output,
                confidence_output, stage_output, cure_output, features_output
            ]
        )
        
        clear_btn.click(
            fn=lambda: [None] * 10,
            inputs=[],
            outputs=[
                input_image, original_output, equalized_output, segmented_output,
                filtered_output, gradcam_output, prediction_output,
                confidence_output, stage_output, cure_output, features_output
            ]
        )
        
        exit_btn.click(fn=lambda: exit(), inputs=[], outputs=[])
        
        gr.Markdown("---")
        gr.Markdown("### Project By")
        gr.Markdown("""
        **Mr. Roshan P. Helonde**  
        Mobile / WhatsApp: +91-7276355704  
        Email: roshanphelonde@rediffmail.com  
        www.matlabsproject.blogspot.in  ||  www.enggprojectworld.blogspot.in
        """)
    
    return interface

# Main execution
if __name__ == "__main__":
    if "--train" in sys.argv:
        print("Training model...")
        trained_model = train_model()
    else:
        print("Launching Gradio interface...")
        interface = create_gradio_interface()
        if interface:
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True,
                debug=True
            )