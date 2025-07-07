import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
from typing import List, cast
from pathlib import Path
import io, sys, os

# Add the parent directory to sys.path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_model

# Initialize Fast Api app 
app = FastAPI()

# CORS middleware setup 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model 
num_classes = 2
class_names = ["NORMAL", "PNEUMONIA"]


# Model is in the parent directory (backend), not in src
MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pth"
model = get_model(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
model.eval()

# Data transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Prediction Function 
def predict_image(img_bytes: bytes) -> str: 
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # transform converts PIL Image to torch.Tensor
    tensor_image = transform(image)  
    # Cast to Tensor for type checking
    tensor_image = cast(torch.Tensor, tensor_image)
    tensor_image = tensor_image.unsqueeze(0).to(device)

    with torch.no_grad(): 
        outputs = model(tensor_image)
        _, predicted = torch.max(outputs, 1)
        predicted_idx = int(predicted.item())
        return class_names[predicted_idx]

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Api routes for prediction 
@app.post("/predict")
async def predict(file: UploadFile = File(...)): 
    try: 
        image_bytes = await file.read()
        prediction = predict_image(image_bytes)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e: 
        return JSONResponse(content={"error": str(e)}, status_code=500)