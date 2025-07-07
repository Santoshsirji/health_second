import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import cast, List
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_model

# Get the absolute path to the script directory and model file
SCRIPT_DIR = Path(__file__).resolve().parent
# Model is likely in the parent directory (backend) rather than in src
MODEL_PATH = SCRIPT_DIR.parent / "model.pth"

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 2

# Class labels (index 0 = NORMAL, index 1 = PNEUMONIA)
class_names: List[str] = ['NORMAL', 'PNEUMONIA']

# Data transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load model
model = get_model(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(str(MODEL_PATH)))
model.eval()

# Inference function
def predict_image(image_path: str) -> str:
    # Load image
    pil_image = Image.open(image_path).convert('RGB')

    # Apply transforms (converts PIL Image to tensor)
    # The transform converts PIL Image to torch.Tensor
    tensor_image = transform(pil_image)  # Now a tensor of shape [3, 224, 224]
    
    # Add batch dimension: [3, 224, 224] → [1, 3, 224, 224]
    # We know tensor_image is a torch.Tensor at this point
    tensor_image = cast(torch.Tensor, tensor_image)
    tensor_image = tensor_image.unsqueeze(0).to(device)

    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass
        outputs = model(tensor_image)

        # Get predicted class
        _, predicted = torch.max(outputs, 1)
        predicted_idx = int(predicted.item())  # Convert to int explicitly
        predicted_class = class_names[predicted_idx]

    return predicted_class

# Example usage
if __name__ == "__main__":
    # Use an absolute path for the sample image too
    sample_image = SCRIPT_DIR.parent / "data" / "chest_xray" / "test" / "NORMAL" / "NORMAL2-IM-0381-0001.jpeg"
    result = predict_image(str(sample_image))
    print(f'Prediction: {result}')
