import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the classes
CLASSES = ['apple', 'banana', 'orange']

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Smaller size for faster processing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 32 * 32, 3)
        
        # Initialize with fixed weights
        torch.manual_seed(42)
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv.bias)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc(x)
        return x

# Global model variable
_model = None

def get_model():
    """Get or initialize the model."""
    global _model
    if _model is None:
        _model = SimpleNet()
        _model.eval()
    return _model

def predict_image(image_path):
    """Predict the class of an image."""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        model = get_model()
        
        with Image.open(image_path).convert('RGB') as image:
            image_tensor = transform(image).unsqueeze(0)
        
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