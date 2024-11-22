from http.server import BaseHTTPRequestHandler
import torch
import json
import base64
import io
import sys
import traceback
from PIL import Image
import numpy as np
from model.model import NeuralNetwork
import torchvision.transforms as transforms

def process_image(image_data):
    try:
        # Decode base64 image
        # Remove data URL prefix if exists
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Resize and transform
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset normalization
        ])
        
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        print(f"Image Processing Error: {e}")
        print(traceback.format_exc())
        raise

def load_model():
    try:
        model = NeuralNetwork()
        model.load_state_dict(torch.load('model/ml_with_pytorch_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Model Loading Error: {e}")
        print(traceback.format_exc())
        raise

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Enable error tracking
        sys.stderr = sys.stdout

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            image_data = data.get('image')
            
            if not image_data:
                print("No image data provided")
                self.send_error(400, "No image data provided")
                return
            
            # Process image
            image_tensor = process_image(image_data)
            
            # Load model and make prediction
            model = load_model()
            with torch.no_grad():
                output = model(image_tensor)
                predicted = output[0].argmax(0).item()
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({"prediction": str(predicted)})
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            print(traceback.format_exc())
            self.send_error(500, str(e))
            return
