import numpy as np
from PIL import Image

def preprocess_image(image, img_size=256):
    """
    Preprocess image for semantic segmentation model.
    Accepts PIL image or numpy array.
    """
    # Convert numpy array to PIL image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Convert to RGB (in case of RGBA or grayscale)
    image = image.convert("RGB")
    
    # Resize image to model input size
    image = image.resize((img_size, img_size))
    
    # Convert back to numpy array
    img_array = np.array(image)
    
    # Normalize image to range [0,1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Expand dimensions for prediction (batch dimension)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    return image, img_tensor
