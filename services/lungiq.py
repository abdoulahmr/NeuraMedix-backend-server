# === 7. Grad-CAM Utility ===
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

def generate_gradcam(wrapped_model, image_tensor, target_class_idx):
    """
    Generate a Grad-CAM heatmap for ConvNeXt model inside WrappedModel.
    Returns a base64-encoded PNG image of the heatmap overlay.
    """
    model = wrapped_model.model
    device = wrapped_model.device
    model.eval()
    image_tensor = image_tensor.to(device)

    activations = None
    gradients = None

    # Register forward and backward hooks on last block of features
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    score = output[0, target_class_idx]
    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = activations.detach()[0]
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]
    heatmap = activations.mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    # Upsample heatmap to input image size
    import cv2
    heatmap = cv2.resize(heatmap, (224, 224))

    # Prepare original image (undo normalization)
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1)

    # Create heatmap overlay
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()

    return base64.b64encode(img_bytes).decode('utf-8')


# === 1. Model Wrapper ===
class WrappedModel:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            return dict(zip(self.class_names, probs))


# === 2. Model Loader Utilities ===
def safe_load_state_dict(path):
    state = torch.load(path, map_location="cpu")
    return state["model"] if "model" in state else state

def load_convnext(num_classes, path):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(safe_load_state_dict(path))
    return model

def load_efficientnet(num_classes, path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(safe_load_state_dict(path))
    return model

def load_resnet(num_classes, path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(safe_load_state_dict(path))
    return model


# === 3. Model Setup ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "lungiq")

# Correct class order: ['No Cancer', 'Cancer']
model1_convnext = WrappedModel(load_convnext(2, os.path.join(MODEL_DIR, "stage_1_convnext.pth")), ["No Cancer", "Cancer"])
model1_efficientnet = WrappedModel(load_efficientnet(2, os.path.join(MODEL_DIR, "stage_1_efficientnet.pth")), ["No Cancer", "Cancer"])
model1_resnet = WrappedModel(load_resnet(2, os.path.join(MODEL_DIR, "stage_1_resnet.pth")), ["No Cancer", "Cancer"])

# Stage 2: Benign vs Malignant
model2_convnext = WrappedModel(load_convnext(2, os.path.join(MODEL_DIR, "stage_2_convnext.pth")), ["Benign", "Malignant"])
model2_efficientnet = WrappedModel(load_efficientnet(2, os.path.join(MODEL_DIR, "stage_2_efficientnet.pth")), ["Benign", "Malignant"])
model2_resnet = WrappedModel(load_resnet(2, os.path.join(MODEL_DIR, "stage_2_resnet.pth")), ["Benign", "Malignant"])

# Stage 3: Subtype (including 'Other')
model3 = WrappedModel(load_convnext(4, os.path.join(MODEL_DIR, "stage_3.pth")), [
    "Adenocarcinoma", "Squamous Cell Carcinoma", "Large Cell Carcinoma", "Other"
])


# === 4. Preprocessing ===
def preprocess(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# === 5. Utilities ===
def percentify(score_dict):
    return {k: f"{v * 100:.1f}%" for k, v in score_dict.items()}

def determine_final_path(cancer, stage2, subtype):
    if cancer["Cancer"] <= 0.5:
        return "No Cancer"
    if stage2["Benign"] > stage2["Malignant"]:
        return "Cancer → Benign"
    top_subtype = max(subtype, key=subtype.get)
    return f"Cancer → Malignant → {top_subtype}"

def average_probabilities(*score_dicts):
    keys = score_dicts[0].keys()
    avg_scores = {k: 0.0 for k in keys}
    n = len(score_dicts)
    for d in score_dicts:
        for k in keys:
            avg_scores[k] += d[k] / n
    return avg_scores


# === 6. Expert System Endpoint ===
def analyze_ct_slice(image_file):
    image_tensor = preprocess(image_file)

    # Step 1: Cancer Detection
    cancer_scores = average_probabilities(
        model1_convnext.predict_proba(image_tensor),
        model1_efficientnet.predict_proba(image_tensor),
        model1_resnet.predict_proba(image_tensor)
    )

    # Step 2: Benign vs Malignant
    if cancer_scores["Cancer"] > 0.5:
        stage2_scores = average_probabilities(
            model2_convnext.predict_proba(image_tensor),
            model2_efficientnet.predict_proba(image_tensor),
            model2_resnet.predict_proba(image_tensor)
        )
    else:
        stage2_scores = {"Benign": 0.0, "Malignant": 0.0}

    # Step 3: Subtype
    if stage2_scores.get("Malignant", 0.0) > 0.5:
        subtype_scores = model3.predict_proba(image_tensor)
    else:
        subtype_scores = {
            "Adenocarcinoma": 0.0,
            "Squamous Cell Carcinoma": 0.0,
            "Large Cell Carcinoma": 0.0,
            "Other": 0.0
        }

    decision = determine_final_path(cancer_scores, stage2_scores, subtype_scores)

    # Grad-CAM for Cancer detection (ConvNeXt, most likely class)
    top_class = max(cancer_scores, key=cancer_scores.get)
    top_class_idx = model1_convnext.class_names.index(top_class)
    gradcam_b64 = generate_gradcam(model1_convnext, image_tensor, top_class_idx)

    return {
        "Step 1: Cancer Detection": percentify(cancer_scores),
        "Step 2: Benign vs Malignant": percentify(stage2_scores),
        "Step 3: Malignant Subtype": percentify(subtype_scores),
        "Final Decision": decision,
        "GradCAM": gradcam_b64
    }
