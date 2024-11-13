from flask import Flask, request, render_template, redirect, url_for
import os
import torch
import torch.nn as nn
from torchvision import transforms
import efficientnet_pytorch as en
from PIL import Image
import numpy as np
from brisque import BRISQUE  # Import BRISQUE for image quality assessment

app = Flask(__name__)

# Define data transforms for image preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ProgressiveNeuralNetwork(nn.Module):
    def __init__(self):
        super(ProgressiveNeuralNetwork, self).__init__()
        self.efficientnet = en.EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, 1)

    def forward(self, x):
        x = self.efficientnet(x)
        return x

# Load the trained model
model = ProgressiveNeuralNetwork()
model.load_state_dict(torch.load('messidor_model.pth'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Save the uploaded image
        image_path = os.path.join('static/uploads', file.filename)
        file.save(image_path)

        # Image processing and prediction code...
        image = Image.open(file)
        image_rgb = image.convert('RGB')  # Ensure the image is in RGB format
        image_tensor = data_transforms(image_rgb)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Get prediction from the model
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.round(torch.sigmoid(output)).item()
        
        diagnosis = "Chance of developing Diabetic Retinopathy" if prediction == 1 else "No Diabetic Retinopathy"

        # BRISQUE score calculation
        brisque_score = calculate_brisque(image_rgb)

        # Check BRISQUE score and assign image quality
        if brisque_score > 20:
            # If the image is low quality, redirect to error page
            return redirect(url_for('error'))

        # If BRISQUE score is acceptable, show the prediction
        return render_template('result.html', 
                               prediction=diagnosis, 
                               image_url=file.filename, 
                               brisque_score=brisque_score)

    else:
        return redirect(request.url)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_brisque(image):
    """Calculate the BRISQUE score for an image."""
    try:
        brisque_obj = BRISQUE(url=False)
        brisque_score = brisque_obj.score(img=np.array(image))  # Convert image to numpy array
        return brisque_score
    except Exception as e:
        return f"Error calculating BRISQUE score: {e}"

@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
