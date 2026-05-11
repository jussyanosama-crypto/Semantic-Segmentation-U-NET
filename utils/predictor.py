import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from .preprocessing import preprocess_image

# Custom loss functions and metrics
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

class SegmentationModel:
    def __init__(self, model_path="model/semantic_segmentation_model.keras"):
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        # Support CPU-only environments by limiting device visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
        
        # Define custom objects dictionary
        custom_objects = {
            "dice_coef": dice_coef,
            "dice_loss": dice_loss,
            "combined_loss": combined_loss
        }
        
        # Load the model with custom objects
        try:
            model = tf.keras.models.load_model(
                self.model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")

    def predict(self, image, threshold=0.6, img_size=256):
        """
        Run inference on the given image.
        Returns the binary mask and resized mask matching original image size.
        """
        # Determine original dimensions
        if isinstance(image, np.ndarray):
            original_h, original_w = image.shape[:2]
        else:
            original_w, original_h = image.size
            
        # Preprocess the image
        _, img_tensor = preprocess_image(image, img_size=img_size)
        
        # Run inference
        pred_mask = self.model.predict(img_tensor, verbose=0)[0]
        
        # Apply thresholding
        binary_mask = (pred_mask > threshold).astype(np.uint8)
        
        # Remove small noise using morphology operations
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Resize mask to match original image size
        resized_mask = cv2.resize(
            binary_mask, 
            (original_w, original_h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Expand dims if it's lost during operations
        if len(resized_mask.shape) == 2:
            resized_mask = np.expand_dims(resized_mask, axis=-1)
            
        return binary_mask, resized_mask
