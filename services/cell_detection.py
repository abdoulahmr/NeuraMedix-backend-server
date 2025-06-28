import matplotlib
matplotlib.use('Agg')

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import io
import base64

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, "cell_detection.pth")

def load_cell_detection_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please ensure 'models/cell_detection.pth' exists.")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def detect_cells(image_filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_filename = os.path.basename(image_filepath)

    try:
        model = load_cell_detection_model(MODEL_PATH, device)
    except FileNotFoundError as e:
        return {
            'cell_count': 0,
            'processed_image': None,
            'original_filename': original_filename,
            'error': f"Model loading failed: {e}"
        }

    try:
        image = Image.open(image_filepath).convert("RGB")
    except Exception as e:
        return {
            'cell_count': 0,
            'processed_image': None,
            'original_filename': original_filename,
            'error': f"Image processing failed: {e}"
        }

    transform = T.ToTensor()
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    cell_count = 0
    filtered_boxes = []
    filtered_scores = []
    for box, score in zip(prediction['boxes'], prediction['scores']):
        if score > 0.5:
            cell_count += 1
            filtered_boxes.append(box.cpu().numpy())
            filtered_scores.append(score.cpu().numpy())

    processed_image_base64 = None
    try:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_tensor.permute(1, 2, 0).cpu().numpy())

        for box, score in zip(filtered_boxes, filtered_scores):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        processed_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        processed_image_base64 = None

    return {
        'cell_count': cell_count,
        'processed_image': processed_image_base64,
        'original_filename': original_filename
    }
