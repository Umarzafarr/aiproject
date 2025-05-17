import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import io

# Define the classes
CLASSES = ['apple', 'banana', 'orange']

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 3)  # 3 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Global model variable
_model = None

def get_model():
    """Get or initialize the model."""
    global _model
    if _model is None:
        _model = SimpleCNN()
        model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        if os.path.exists(model_path):
            try:
                # Load with newer PyTorch version compatibility
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                # Handle potential version differences
                if isinstance(state_dict, dict):
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                _model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                # Initialize with random weights if loading fails
                _model = SimpleCNN()
        _model.eval()
    return _model

def predict_image(image_path):
    """
    Predict the class of an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Predicted class name
    """
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load the model
        model = get_model()
        
        # Open and preprocess the image
        with Image.open(image_path).convert('RGB') as image:
            # Ensure image is in RGB mode
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = CLASSES[predicted.item()]
            
        return predicted_class
        
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    try:
        prediction = predict_image(image_path)
        print(f"Predicted class: {prediction}")
    except Exception as e:
        print(f"Error: {str(e)}") 