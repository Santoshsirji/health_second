import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_model  # your ChestXrayCNN loader

# Get the absolute path to the data directory
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "chest_xray"

# Device config (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
num_classes = 2

# Data transforms (resize image, convert to tensor, normalize pixel values)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load validation dataset using absolute path
val_dataset = datasets.ImageFolder(root=str(DATA_DIR / "val"), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = get_model(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set model to evaluation mode

# Validation function
def validate():
    model.eval()  # Set model to evaluation mode
    total = 0
    correct = 0
    
    # Disable gradient calculation for validation
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            
            # Update stats
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    
    # Calculate class-wise accuracy
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Print class-wise accuracy
    for i in range(num_classes):
        class_name = val_dataset.classes[i]
        class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'Accuracy of {class_name}: {class_acc:.2f}%')

# Run validation
if __name__ == "__main__":
    validate()