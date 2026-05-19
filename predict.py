from PIL import Image
from src.inference.predictor import Predictor

# 1. Initialize the Predictor
# This will automatically load the backbone and crop heads from your "models" directory
predictor = Predictor(model_dir="models")

# 2. Open an image using PIL (Pillow)
image_path = "<path>"
image = Image.open(image_path)

# 3. Call the predict method
# Make sure the crop_type matches one of the crops you have trained models for
try:
    result = predictor.predict(image=image, crop_type="<type>")
    
    print(f"Predicted Disease: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"All Probabilities: {result['all_probabilities']}")
    print(f"Model version: {result['model_version']}")
    
except ValueError as e:
    print(f"Error: {e}")
