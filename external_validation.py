"""
External Dataset Validation for Lung Cancer Detection Model
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# CONFIGURATION
# ----------------------------

MODEL_PATH = "models/lung_cancer_model.pth"  # Or try "models/omnighost_lung_best.pth"
DATASET_PATH = "datasets/external_dataset"
IMAGE_SIZE = 224
BATCH_SIZE = 16
CLASS_NAMES = ["normal", "benign", "malignant"]
RESULTS_DIR = "evaluation_results"


# ----------------------------
# MODEL DEFINITION (Must match training architecture)
# ----------------------------

class LungCancerModel(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCancerModel, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(weights=None)
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


# ----------------------------
# DATASET
# ----------------------------

class ExternalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# ----------------------------
# LOAD DATASET
# ----------------------------

def load_external_dataset():
    image_paths = []
    labels = []
    
    class_map = {
        "normal": 0,
        "benign": 1,
        "malignant": 2
    }
    
    print(f"\nScanning dataset at: {DATASET_PATH}")
    
    for class_name, label in class_map.items():
        folder = os.path.join(DATASET_PATH, class_name)
        
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue
            
        files = [f for f in os.listdir(folder) 
                if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        print(f"📁 {class_name}: {len(files)} images found")
        
        for file in files:
            image_paths.append(os.path.join(folder, file))
            labels.append(label)
    
    return image_paths, labels


# ----------------------------
# MODEL LOADING
# ----------------------------

def load_model(device, model_path=MODEL_PATH):
    print(f"\n📥 Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with correct architecture
    model = LungCancerModel(num_classes=3)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        val_acc = checkpoint.get('val_accuracy', 'Unknown')
        epoch = checkpoint.get('epoch', 'Unknown')
        print(f"✅ Model loaded! Validation accuracy: {val_acc}%, Epoch: {epoch}")
    else:
        # Try loading directly (might be state_dict only)
        try:
            model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully (direct state_dict)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    model.to(device)
    model.eval()
    return model


# ----------------------------
# PLOT FUNCTIONS
# ----------------------------

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 12}
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix - External Validation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc(y_true, y_prob, save_path):
    plt.figure(figsize=(8, 6))
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    for i, class_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - External Validation", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_class_distribution(labels, save_path):
    """Plot class distribution in external dataset"""
    counts = [sum(1 for l in labels if l == i) for i in range(len(CLASS_NAMES))]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(CLASS_NAMES, counts, color=['#2ecc71', '#f39c12', '#e74c3c'])
    plt.title("Class Distribution - External Dataset", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Images", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ----------------------------
# EVALUATION
# ----------------------------

def evaluate(model_path=MODEL_PATH):
    print("\n" + "="*60)
    print("🔬 EXTERNAL DATASET VALIDATION")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(RESULTS_DIR, f"external_eval_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {save_dir}")
    
    # Load model
    try:
        model = load_model(device, model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load external dataset
    print("\n📊 Loading external dataset...")
    image_paths, labels = load_external_dataset()
    
    if len(image_paths) == 0:
        print("❌ No images found in external dataset!")
        return
    
    print(f"\n📈 Total external images: {len(image_paths)}")
    
    # Plot class distribution
    plot_class_distribution(labels, os.path.join(save_dir, "class_distribution.png"))
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = ExternalDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Evaluation
    print("\n🔍 Running inference on external dataset...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels_batch) in enumerate(loader):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches")
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )
    
    # Print results
    print("\n" + "="*60)
    print("🎯 EXTERNAL VALIDATION RESULTS")
    print("="*60)
    
    print(f"\n📊 ACCURACY: {accuracy:.2f}%\n")
    
    print("📋 CLASSIFICATION REPORT:")
    print(report)
    
    print("\n🎭 CONFUSION MATRIX:")
    print("True \\ Predicted\tNormal\tBenign\tMalignant")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:15}\t{cm[i][0]}\t{cm[i][1]}\t{cm[i][2]}")
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    
    # Confusion Matrix
    plot_confusion_matrix(cm, os.path.join(save_dir, "confusion_matrix.png"))
    
    # ROC Curves
    plot_roc(all_labels, all_probs, os.path.join(save_dir, "roc_curves.png"))
    
    # Save detailed predictions
    df = pd.DataFrame({
        "image_path": image_paths,
        "true_label": [CLASS_NAMES[i] for i in all_labels],
        "predicted_label": [CLASS_NAMES[i] for i in all_preds],
        "correct": [true == pred for true, pred in zip(all_labels, all_preds)],
        "normal_prob": [prob[0] for prob in all_probs],
        "benign_prob": [prob[1] for prob in all_probs],
        "malignant_prob": [prob[2] for prob in all_probs]
    })
    
    df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    
    # Save summary
    with open(os.path.join(save_dir, "evaluation_summary.txt"), "w") as f:
        f.write("EXTERNAL DATASET VALIDATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Validation Timestamp: {timestamp}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"External Dataset Path: {DATASET_PATH}\n")
        f.write(f"Total Images: {len(image_paths)}\n\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"\n✅ VALIDATION COMPLETED SUCCESSFULLY!")
    print(f"📁 Results saved in: {save_dir}")
    
    # Performance comparison
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   Internal Validation: ~96.82%")
    print(f"   External Validation: {accuracy:.2f}%")
    print(f"   Gap: {96.82 - accuracy:.2f}%")
    
    if accuracy >= 90:
        print("   ✅ Excellent generalization!")
    elif accuracy >= 80:
        print("   ⚠️ Good generalization, but some domain shift")
    elif accuracy >= 70:
        print("   ⚠️ Moderate generalization - consider domain adaptation")
    else:
        print("   ❌ Poor generalization - significant domain shift detected")
    
    return accuracy, cm, report


# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='External validation for lung cancer model')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Try both model files if specified one doesn't work
    model_files = [args.model, "models/omnighost_lung_best.pth"]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"\n🔄 Trying model: {model_file}")
            try:
                evaluate(model_file)
                break
            except Exception as e:
                print(f"❌ Failed with {model_file}: {e}")
                continue