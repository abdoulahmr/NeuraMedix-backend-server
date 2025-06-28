import torch
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b2
from PIL import Image
import io
import base64
import os

# Load once globally
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(BASE_DIR, 'models/efficientnet_tumor_classifier_92.3.pth')

# Define class labels (adjust if needed)
class_names = ['No Tumor', 'Tumor']

# Load and configure model
model = efficientnet_b2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transform image to match model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_lung_cancer_from_image(image_bytes):
    try:
        # Load and transform
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            predicted_index = np.argmax(probabilities)
            prediction = class_names[predicted_index]
            probability = float(probabilities[predicted_index]) * 100

        # Optionally return annotated image as base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return prediction, probability, encoded_image

    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
