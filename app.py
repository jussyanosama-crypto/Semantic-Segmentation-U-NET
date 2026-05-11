import os
import cv2
import numpy as np
from PIL import Image
from utils.predictor import SegmentationModel
from utils.visualization import create_overlay, visualize_results

def main():
    print("🚀 Initializing Semantic Segmentation Local Testing Script...")
    
    # Check if demo image exists
    demo_path = "assets/demo.png"
    if not os.path.exists(demo_path):
        print(f"❌ Error: Demo image not found at {demo_path}")
        print("Please place a test image at assets/demo.png to run this script.")
        return
        
    print(f"📦 Loading demo image from {demo_path}...")
    try:
        image = Image.open(demo_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Load model
    print("🧠 Loading U-Net segmentation model...")
    try:
        model = SegmentationModel()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
        
    # Run prediction
    print("⚙️ Running inference (this may take a moment)...")
    try:
        binary_mask, resized_mask = model.predict(image, threshold=0.6)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return
        
    print("🎨 Generating visualizations...")
    # Create overlay
    overlay_output = create_overlay(image, resized_mask, alpha=0.5)
    
    # Prepare save directories if they don't exist
    os.makedirs("output", exist_ok=True)
    
    # Save predictions
    # Convert mask to 255 for saving
    save_mask = (resized_mask * 255).astype(np.uint8)
    
    # Save predicted mask
    cv2.imwrite("output/predicted_mask.png", save_mask)
    
    # Save overlay output (convert RGB to BGR for OpenCV saving)
    overlay_bgr = cv2.cvtColor(overlay_output, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output/overlay_output.png", overlay_bgr)
    
    print("✅ Inference Successful!")
    print("Saved results to 'output' directory:")
    print("  - output/predicted_mask.png")
    print("  - output/overlay_output.png")
    
if __name__ == "__main__":
    main()
