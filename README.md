# Chest X-Ray Pneumonia Detection API

A deep learning API that detects pneumonia from chest X-ray images using PyTorch and FastAPI.

## Project Structure

```
health_second/
├── backend/
│   ├── data/
│   │   └── chest_xray/       # Dataset with train, test, and val sets
│   ├── src/
│   │   ├── api.py            # FastAPI application
│   │   ├── inference.py      # Inference script
│   │   ├── model.py          # Model architecture
│   │   ├── train.py          # Training script
│   │   └── validate.py       # Validation script
│   └── model.pth             # Trained model weights
├── requirements.txt          # Python dependencies
└── render.yaml               # Render deployment configuration
```

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the API locally:
   ```
   cd backend/src
   uvicorn api:app --reload
   ```

3. Access the API at http://127.0.0.1:8000

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Upload a chest X-ray image for pneumonia detection

## Deploying to Render

1. Push your code to GitHub
2. Connect your GitHub repository to Render
3. Render will automatically detect the `render.yaml` file and create the service
4. The API will be deployed at `https://chest-xray-api.onrender.com`

## Model Information

- Architecture: Custom CNN for chest X-ray classification
- Classes: NORMAL, PNEUMONIA
- Input: 224x224 RGB images
- Training dataset: ~5,000 chest X-ray images 