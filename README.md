# U-Net Semantic Segmentation with MobileNetV2

A deep learning project for binary semantic segmentation using a **U-Net architecture with a pretrained MobileNetV2 encoder**. The project covers transfer learning, custom loss functions, model training, evaluation, image segmentation, post-processing, and interactive deployment with Streamlit.

## Project Overview

The goal is to build a semantic segmentation model that identifies the target region at the **pixel level** rather than simply classifying the entire image.

Unlike object detection, which predicts bounding boxes around objects, semantic segmentation assigns a class label to individual pixels.

The project uses **U-Net with MobileNetV2** as the encoder to combine strong image feature extraction with precise segmentation.

## Model Architecture

The model is based on a U-Net architecture with:

* **MobileNetV2** as the pretrained encoder
* Transfer learning from ImageNet pretrained weights
* U-Net decoder for recovering spatial information
* Skip connections between encoder and decoder layers
* Binary segmentation output

The pretrained MobileNetV2 encoder provides useful visual features while the U-Net decoder reconstructs a detailed segmentation mask.

## Loss Function and Metrics

Because segmentation quality depends on both pixel-level classification and overlap with the target region, the project uses a combined loss:

**Combined Loss = Binary Cross-Entropy + Dice Loss**

The project also uses the **Dice Coefficient** as a segmentation metric.

### Dice Coefficient

Dice measures the overlap between the predicted mask and the ground-truth mask:

$$
Dice = \frac{2|Prediction \cap GroundTruth|}
{|Prediction| + |GroundTruth|}
$$

A higher Dice score indicates better overlap between the predicted and true segmentation regions.

## Training

The model was trained for up to **100 epochs** with:

* Batch size: 8
* Early stopping
* Model checkpointing
* Validation monitoring
* Best model weight restoration

Early stopping was configured with a patience of **15 epochs**, while the best model checkpoint was saved during training.

The model reached a validation Dice score of approximately **0.93+** during training, with the highest observed validation Dice around **0.937**.

> The checkpoint was selected according to validation loss rather than selecting the epoch solely by Dice score.

## Inference Pipeline

The deployment pipeline performs the following steps:

1. Load the trained U-Net model.
2. Load and preprocess the input image.
3. Resize the image to the model input size.
4. Generate a pixel-level prediction.
5. Apply a confidence threshold to obtain a binary mask.
6. Apply morphological opening and closing to reduce small noise.
7. Resize the predicted mask back to the original image dimensions.
8. Generate an overlay showing the predicted segmentation on the original image.

## Streamlit Application

A Streamlit application is included for interactive inference.

The application allows users to:

* Upload their own image.
* Use the included demo image.
* Adjust the segmentation confidence threshold.
* Adjust overlay transparency.
* Generate a predicted segmentation mask.
* View the binary mask.
* View the segmentation overlay on the original image.

### Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application loads the trained model from:

```text
model/semantic_segmentation_model.keras
```

## Project Structure

```text
Semantic-Segmentation-U-NET/
│
├── assets/
│   └── demo.png
│
├── model/
│   └── semantic_segmentation_model.keras
│
├── utils/
│   ├── predictor.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── packages.txt
└── README.md
```

## Technologies

* Python
* TensorFlow / Keras
* U-Net
* MobileNetV2
* Transfer Learning
* OpenCV
* NumPy
* Pillow
* Streamlit
* Matplotlib

## Key Takeaways

This project demonstrates an end-to-end semantic segmentation workflow, including:

* Transfer learning with a pretrained CNN encoder.
* U-Net architecture for pixel-level segmentation.
* Custom Dice-based loss and evaluation.
* Binary segmentation.
* Early stopping and model checkpointing.
* Image preprocessing and post-processing.
* Morphological operations for mask refinement.
* Interactive model inference through Streamlit.

## Limitations

The model's performance depends on the quality and diversity of the training and validation data.

The validation Dice score is a useful measure of segmentation overlap, but it does not guarantee equally strong performance on images from different distributions.

Further evaluation on a larger and more diverse test set would provide a stronger estimate of real-world generalization.

## Future Improvements

Possible improvements include:

* Evaluating the model on a dedicated unseen test set.
* Increasing dataset size and diversity.
* Experimenting with different segmentation architectures.
* Performing additional augmentation and hyperparameter tuning.
* Comparing different pretrained encoders.
* Adding more detailed segmentation metrics such as IoU.
* Improving the deployment interface with additional visualization and batch-processing options.
