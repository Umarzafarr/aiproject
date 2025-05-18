import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Define the classes
CLASSES = ['apple', 'banana', 'orange']

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class CustomResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomResNet, self).__init__()
        # Load pre-trained ResNet model
        self.model = models.resnet18(pretrained=False)
        
        # Modify the final layer for our number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model():
    model = CustomResNet(num_classes=len(CLASSES))
    # Initialize model weights deterministically
    torch.manual_seed(42)
    for param in model.parameters():
        if len(param.shape) > 1:  # For weight matrices
            nn.init.xavier_uniform_(param)
        else:  # For bias vectors
            nn.init.zeros_(param)
    model.eval()
    return model

# Global model variable
_model = None

def get_model():
    """Get or initialize the model."""
    global _model
    if _model is None:
        _model = load_model()
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