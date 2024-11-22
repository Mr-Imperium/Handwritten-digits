import torch
import json
import base64
import io
import traceback
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from model.model import NeuralNetwork

def load_model():
    try:
        model = NeuralNetwork()
        # Use absolute path or ensure model is in the correct deployment location
        model.load_state_dict(torch.load('/opt/conda/ml_with_pytorch_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Model Loading Error: {e}")
        print(traceback.format_exc())
        raise

def process_image(image_data):
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        print(f"Image Processing Error: {e}")
        print(traceback.format_exc())
        raise

def handler(request):
    try:
        # Vercel's request handling
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "No image data provided"})
            }
        
        # Process image
        image_tensor = process_image(image_data)
        
        # Load model and make prediction
        model = load_model()
        with torch.no_grad():
            output = model(image_tensor)
            predicted = output[0].argmax(0).item()
        
        return {
            'statusCode': 200,
            'body': json.dumps({"prediction": str(predicted)})
        }
    
    except Exception as e:
        print(f"Prediction Error: {e}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
