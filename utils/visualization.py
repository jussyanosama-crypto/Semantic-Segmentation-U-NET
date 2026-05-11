import cv2
import numpy as np
from PIL import Image

def create_overlay(image, mask, alpha=0.5, color=(0, 255, 0)):
    """
    Generate clean overlay visualization.
    Returns overlay image as numpy array.
    """
    # Ensure image is numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        # Assume numpy array, convert to RGB if grayscale
        if len(image.shape) == 2:
            img_array = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_array = image.copy()
            
    # Ensure mask is 2D for masking operations
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask
        
    # Ensure mask is uint8 and binary (0 and 1)
    if mask_2d.max() > 1:
        mask_2d = (mask_2d > 0).astype(np.uint8)
        
    # Create colored mask
    colored_mask = np.zeros_like(img_array)
    colored_mask[mask_2d == 1] = color
    
    # Apply overlay where mask is 1
    overlay = img_array.copy()
    mask_indices = mask_2d == 1
    
    # Smooth blending
    overlay[mask_indices] = cv2.addWeighted(
        img_array[mask_indices], 1 - alpha, 
        colored_mask[mask_indices], alpha, 
        0
    )
    
    return overlay

def visualize_results(original, mask, overlay):
    """
    Returns a combined image showing original, mask, and overlay side-by-side.
    """
    import matplotlib.pyplot as plt
    import io
    
    # Create a figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    if isinstance(original, Image.Image):
        axes[0].imshow(original)
    else:
        axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Mask
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        axes[1].imshow(mask[:, :, 0], cmap='gray')
    else:
        axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay Result")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    # Save to a bytes buffer instead of displaying
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    # Read image from buffer
    combined_img = Image.open(buf)
    return np.array(combined_img)
