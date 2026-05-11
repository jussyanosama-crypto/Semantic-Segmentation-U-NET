import streamlit as st
import numpy as np
from PIL import Image
import os

from utils.predictor import SegmentationModel
from utils.visualization import create_overlay

# Page configuration
st.set_page_config(
    page_title="Semantic Segmentation",
    page_icon="🧠",
    layout="centered"
)

# Initialize model lazily using st.cache_resource
@st.cache_resource
def load_model():
    model_path = "model/semantic_segmentation_model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please ensure the model file is placed correctly.")
        return None
    try:
        return SegmentationModel(model_path=model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("🧠 Semantic Segmentation App")
    st.markdown("""
        Upload an image to perform semantic segmentation. 
        The model identifies specific regions and overlays a mask on the original image.
    """)

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Higher values require the model to be more confident to label a pixel."
    )
    
    overlay_alpha = st.sidebar.slider(
        "Overlay Transparency",
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help="Adjust the transparency of the predicted mask overlay."
    )
    
    # Main area
    model = load_model()
    
    if model is None:
        st.stop()

    st.subheader("1. Select Image")
    
    # Image selection mechanism
    upload_option = st.radio("Choose image source:", ("Upload your own", "Use Demo Image"))
    
    image = None
    
    if upload_option == "Upload your own":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error("Invalid image format. Please upload a valid image.")
    else:
        demo_path = "assets/demo.png"
        if os.path.exists(demo_path):
            image = Image.open(demo_path).convert("RGB")
            st.info("Using included demo image.")
        else:
            st.warning("Demo image not found at assets/demo.png. Please upload your own.")

    if image is not None:
        st.image(image, caption="Input Image", use_container_width=True)
        
        st.subheader("2. Run Inference")
        if st.button("Generate Segmentation Mask", type="primary"):
            with st.spinner("Processing image through U-Net model..."):
                try:
                    # Run prediction
                    _, resized_mask = model.predict(image, threshold=confidence_threshold)
                    
                    # Create visualization
                    overlay = create_overlay(image, resized_mask, alpha=overlay_alpha)
                    
                    # Display results in columns
                    st.success("Segmentation complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Convert mask to visual format
                        display_mask = (resized_mask * 255).astype(np.uint8)
                        if len(display_mask.shape) == 3 and display_mask.shape[2] == 1:
                            display_mask = display_mask[:, :, 0]
                        st.image(display_mask, caption="Predicted Mask", use_container_width=True, clamp=True)
                        
                    with col2:
                        st.image(overlay, caption="Overlay Result", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
