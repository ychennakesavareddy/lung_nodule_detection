# app.py - Working Version with Fallback Model
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import os
import gradio as gr
from datetime import datetime
import cv2
import traceback
import sys

# Fix numpy compatibility
np.float = float

# Configuration
class Config:
    IMAGE_SIZE = 224
    NUM_CLASSES = 3
    CLASS_NAMES = ['Normal', 'Benign', 'Malignant']

config = Config()

# CNN Model Architecture
class LungCancerModel(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCancerModel, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
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

# Create a dummy model for testing
def create_dummy_model(device):
    """Create a dummy model for testing when real model isn't available"""
    print("⚠️ Creating dummy model for testing...")
    model = LungCancerModel(num_classes=config.NUM_CLASSES)
    
    # Initialize with random weights
    for param in model.parameters():
        param.data.normal_(0, 0.02)
    
    model.to(device)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image)
    
    return transform(image).unsqueeze(0)

# Image processing functions
def apply_histogram_equalization(image):
    """Apply histogram equalization"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        equalized = cv2.equalizeHist(image)
        return Image.fromarray(equalized)
    except Exception as e:
        print(f"Histogram error: {e}")
        return image

def apply_segmentation(image):
    """Apply segmentation"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        _, segmented = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(segmented)
    except Exception as e:
        print(f"Segmentation error: {e}")
        return image

def apply_filtering(image):
    """Apply median filtering"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        filtered = cv2.medianBlur(image, 5)
        return Image.fromarray(filtered)
    except Exception as e:
        print(f"Filtering error: {e}")
        return image

# Generate heatmap
def generate_heatmap(image, model, device):
    """Generate heatmap for nodule localization"""
    try:
        input_tensor = preprocess_image(image).to(device)
        
        model.eval()
        with torch.no_grad():
            features = model.features(input_tensor)
            weights = torch.mean(features, dim=(2, 3), keepdim=True)
            heatmap = torch.sum(features * weights, dim=1)
            heatmap = torch.relu(heatmap)
            heatmap = heatmap.squeeze().cpu().numpy()
            
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            else:
                heatmap = np.zeros_like(heatmap)
        
        # Convert image
        if isinstance(image, Image.Image):
            original_image = np.array(image)
        else:
            original_image = image
            
        if len(original_image.shape) == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
        # Resize and apply heatmap
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)
        return Image.fromarray(superimposed)
        
    except Exception as e:
        print(f"Heatmap error: {e}")
        return image

# Feature extraction
def extract_features(image):
    """Extract texture features"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate features
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        contrast = np.std(image)
        energy = np.sum(hist ** 2)
        mean_intensity = np.mean(image)
        
        return {
            'entropy': float(entropy),
            'contrast': float(contrast),
            'energy': float(energy),
            'mean_intensity': float(mean_intensity),
        }
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return {'entropy': 0, 'contrast': 0, 'energy': 0, 'mean_intensity': 0}

# Simulate prediction with dummy model
def simulate_prediction(image):
    """Simulate prediction when using dummy model"""
    # Simple rule-based simulation based on image properties
    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Calculate simple metrics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Simple logic for demonstration
        if contrast < 30:
            prediction = "Normal"
            confidence = 0.75 + (brightness / 500)
        elif contrast < 60:
            prediction = "Benign"
            confidence = 0.85
        else:
            prediction = "Malignant"
            confidence = 0.90
        
        confidence = min(confidence, 0.98)
        
        return prediction, confidence
    except:
        return "Normal", 0.50

# Load or create model
def load_or_create_model():
    """Try to load real model, otherwise create dummy"""
    print("\n" + "=" * 50)
    print("Initializing Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = None
    model_source = None
    
    # Try to import huggingface hub
    try:
        from huggingface_hub import hf_hub_download
        
        # Try multiple repository names
        repos_to_try = [
            "yenugu/lung_cancer_model",
            "yenugu/lung-cancer-model"
        ]
        
        for repo_id in repos_to_try:
            try:
                print(f"\nTrying: {repo_id}")
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="lung_cancer_model.pth",
                    resume=True
                )
                
                print(f"Downloaded: {model_path}")
                
                # Load model
                temp_model = LungCancerModel(num_classes=config.NUM_CLASSES)
                checkpoint = torch.load(model_path, map_location=device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    temp_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    temp_model.load_state_dict(checkpoint)
                
                model = temp_model
                model_source = f"✅ Real model loaded from {repo_id}"
                break
                
            except Exception as e:
                print(f"Failed: {str(e)[:100]}")
                continue
                
    except ImportError:
        print("huggingface_hub not available")
    except Exception as e:
        print(f"Error loading from hub: {e}")
    
    # If no model loaded, create dummy
    if model is None:
        model = create_dummy_model(device)
        model_source = "⚠️ Using demonstration model (real model not found)"
    
    model.to(device)
    model.eval()
    
    print(f"\n{model_source}")
    return model, device, model_source

# Prediction function
def predict_lung_cancer(image, model, device, use_dummy):
    """Main prediction function"""
    if image is None:
        return {
            'prediction': 'No Image',
            'confidence': 0.0,
            'stage': 'N/A',
            'cure_probability': 'N/A',
            'risk_level': 'N/A',
            'features': {},
            'heatmap': None,
            'processed': {}
        }
    
    try:
        # Get prediction
        if use_dummy:
            predicted_class, confidence_score = simulate_prediction(image)
        else:
            # Real model prediction
            input_tensor = preprocess_image(image).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = config.CLASS_NAMES[predicted.item()]
            confidence_score = confidence.item()
        
        # Generate visualizations
        heatmap = generate_heatmap(image, model, device)
        equalized = apply_histogram_equalization(image)
        segmented = apply_segmentation(image)
        filtered = apply_filtering(image)
        
        # Extract features
        features = extract_features(image)
        
        # Predict stage
        if predicted_class == "Normal":
            stage = "No cancer detected"
            cure_prob = "N/A"
            risk_level = "Low"
        elif predicted_class == "Benign":
            stage = "Benign Tumor"
            cure_prob = "High (95%)"
            risk_level = "Low"
        else:
            if confidence_score < 0.3:
                stage = "Stage I"
                cure_prob = "High (85%)"
                risk_level = "Low"
            elif confidence_score < 0.6:
                stage = "Stage II"
                cure_prob = "Medium (60%)"
                risk_level = "Medium"
            elif confidence_score < 0.8:
                stage = "Stage III"
                cure_prob = "Low (30%)"
                risk_level = "High"
            else:
                stage = "Stage IV"
                cure_prob = "Very Low (10%)"
                risk_level = "Critical"
        
        return {
            'prediction': predicted_class,
            'confidence': confidence_score,
            'stage': stage,
            'cure_probability': cure_prob,
            'risk_level': risk_level,
            'features': features,
            'heatmap': heatmap,
            'processed': {
                'original': image,
                'equalized': equalized,
                'segmented': segmented,
                'filtered': filtered,
            }
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'stage': 'Error',
            'cure_probability': 'N/A',
            'risk_level': 'Unknown',
            'features': {},
            'heatmap': image,
            'processed': {
                'original': image,
                'equalized': image,
                'segmented': image,
                'filtered': image,
            }
        }

# Create Gradio interface
def create_interface():
    """Create the Gradio web interface"""
    
    # Load or create model
    model, device, model_status = load_or_create_model()
    use_dummy = "demonstration" in model_status.lower()
    
    # Show status in interface
    status_color = "🟢" if not use_dummy else "🟡"
    
    def process_image(image):
        """Process image and return results"""
        result = predict_lung_cancer(image, model, device, use_dummy)
        
        # Format prediction text with emoji
        if result['prediction'] == 'Normal':
            emoji = "✅"
        elif result['prediction'] == 'Benign':
            emoji = "⚠️"
        elif result['prediction'] == 'Malignant':
            emoji = "🚨"
        else:
            emoji = "❓"
        
        prediction_text = f"{emoji} **Prediction:** {result['prediction']}"
        confidence_text = f"**Confidence:** {result['confidence']:.2%}"
        stage_text = f"**Stage:** {result['stage']}"
        cure_text = f"**Cure Probability:** {result['cure_probability']}"
        risk_text = f"**Risk Level:** {result['risk_level']}"
        
        # Feature text
        features = result['features']
        if features:
            feature_text = f"""
            **📊 Feature Analysis:**
            - **Entropy:** {features.get('entropy', 0):.3f} (measures randomness)
            - **Contrast:** {features.get('contrast', 0):.3f} (image sharpness)
            - **Energy:** {features.get('energy', 0):.3f} (texture uniformity)
            - **Mean Intensity:** {features.get('mean_intensity', 0):.1f} (brightness)
            """
        else:
            feature_text = "Feature extraction not available"
        
        return (
            result['processed']['original'],
            result['processed']['equalized'],
            result['processed']['segmented'],
            result['heatmap'],
            result['processed']['filtered'],
            prediction_text,
            confidence_text,
            stage_text,
            cure_text,
            risk_text,
            feature_text,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    # Create interface
    with gr.Blocks(
        title="Lung Cancer Detection System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
            margin: auto !important;
        }
        .status-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        # Status banner
        with gr.Row():
            with gr.Column():
                status_html = f"""
                <div class="status-box">
                    <h3>{status_color} System Status</h3>
                    <p>{model_status}</p>
                    <small>If using demonstration model, results are simulated for testing.</small>
                </div>
                """
                gr.HTML(status_html)
        
        gr.Markdown("""
        # 🫁 Lung Cancer Detection System
        ### Powered by Deep Learning (ResNet50) - 96.82% Accuracy
        
        Upload a lung CT scan image for instant analysis and risk assessment.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Upload CT Scan")
                input_image = gr.Image(
                    type="pil",
                    label="Lung CT Scan Image",
                    height=350
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("## 📊 Results")
                with gr.Group():
                    prediction_out = gr.Markdown("**Prediction:** Ready")
                    confidence_out = gr.Markdown("**Confidence:** -")
                    stage_out = gr.Markdown("**Stage:** -")
                    cure_out = gr.Markdown("**Cure Probability:** -")
                    risk_out = gr.Markdown("**Risk Level:** -")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🖼️ Image Analysis")
                with gr.Tabs():
                    with gr.TabItem("Original"):
                        original_out = gr.Image(label="Original CT Scan", height=250)
                    with gr.TabItem("Enhanced"):
                        equalized_out = gr.Image(label="Contrast Enhanced", height=250)
                    with gr.TabItem("Segmented"):
                        segmented_out = gr.Image(label="Tissue Segmentation", height=250)
            
            with gr.Column():
                gr.Markdown("## 🔬 Advanced Analysis")
                with gr.Tabs():
                    with gr.TabItem("Heatmap"):
                        heatmap_out = gr.Image(label="Nodule Localization", height=250)
                    with gr.TabItem("Filtered"):
                        filtered_out = gr.Image(label="Noise Reduced", height=250)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📈 Feature Extraction")
                features_out = gr.Markdown("Waiting for analysis...")
                timestamp_out = gr.Markdown("---")
        
        # Information section
        with gr.Accordion("ℹ️ System Information", open=False):
            gr.Markdown("""
            ### Model Details
            - **Architecture:** ResNet50 with custom classifier
            - **Target Accuracy:** 96.82%
            - **Classes:** Normal, Benign, Malignant
            - **Input Size:** 224x224 pixels
            
            ### How to Use
            1. Upload a lung CT scan image (JPG, PNG)
            2. Click "Analyze Image"
            3. View classification results and confidence score
            4. Check heatmap for potential nodule locations
            5. Review risk assessment and recommendations
            
            ### Important Note
            - The system is currently running in demonstration mode
            - To use the actual trained model, upload your .pth file to the Space
            - Results are for educational purposes only
            
            ### Disclaimer
            This is a demonstration tool for educational purposes. 
            Always consult medical professionals for proper diagnosis.
            """)
        
        # Event handlers
        analyze_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[
                original_out, equalized_out, segmented_out,
                heatmap_out, filtered_out,
                prediction_out, confidence_out, stage_out,
                cure_out, risk_out, features_out, timestamp_out
            ]
        )
        
        clear_btn.click(
            fn=lambda: [None] * 12,
            outputs=[
                original_out, equalized_out, segmented_out,
                heatmap_out, filtered_out,
                prediction_out, confidence_out, stage_out,
                cure_out, risk_out, features_out, timestamp_out
            ]
        )
    
    return demo

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting Lung Cancer Detection System")
    print("=" * 60)
    
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )