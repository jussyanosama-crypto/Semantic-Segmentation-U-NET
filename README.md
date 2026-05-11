# 🧠 Semantic Segmentation Deployment

A production-ready project scaffold for deploying a U-Net semantic segmentation model using Streamlit Cloud. This repository provides modular inference, robust preprocessing, and interactive visualizations.

## 📁 Folder Structure

```
semantic-segmentation-deployment/
│
├── app.py                 # Local testing script for CLI inference
├── streamlit_app.py       # Streamlit UI application 
├── requirements.txt       # Python package dependencies
├── packages.txt           # OS-level dependencies (e.g. libgl1 for OpenCV)
├── README.md              # Project documentation
│
├── model/
│   └── semantic_segmentation_model.keras  # Pre-trained U-Net model
│
├── utils/
│   ├── predictor.py       # Model loading and inference logic
│   ├── preprocessing.py   # Image normalization and resizing
│   └── visualization.py   # Mask overlay generation functions
│
└── assets/
    └── demo.png           # Sample image for testing
```

## 🧠 Model Description
The model located at `model/semantic_segmentation_model.keras` is a Binary Semantic Segmentation model built using TensorFlow/Keras. It uses a U-Net architecture with a MobileNetV2 pretrained encoder. The model is capable of predicting binary masks on given image inputs.

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run CLI Script
Run the local inference script which processes `assets/demo.png` and outputs the results to a new `output/` folder.
```bash
python app.py
```

### 3. Run Streamlit UI
Start the interactive Streamlit application to upload your own images or test the included demo:
```bash
streamlit run streamlit_app.py
```

## ☁️ Streamlit Cloud Deployment Steps

This application is fully optimized for Streamlit Cloud deployment:
- Uses `opencv-python-headless` and explicitly includes `libgl1` in `packages.txt` for Linux headless environment compatibility.
- Forces CPU mode automatically so it does not fail on CPU-only Streamlit instances.
- Handles custom TensorFlow/Keras objects natively inside the model loader.

**To deploy:**
1. Commit this entire directory to a GitHub repository.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app" and point it to your GitHub repository.
4. Set the "Main file path" to `streamlit_app.py`.
5. Click "Deploy"! The cloud environment will automatically install OS packages from `packages.txt` and Python packages from `requirements.txt`.
