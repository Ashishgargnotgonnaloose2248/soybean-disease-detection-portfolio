# 🌱 Soybean Disease Detection Using Deep Learning

A computer vision project focused on classifying soybean leaf images into six health and disease categories using deep learning, with model evaluation and explainable AI.

## 📌 Project Overview

Soybean crops can be affected by various diseases and pests that impact plant health and productivity. This project explores deep learning-based image classification to identify visible soybean leaf conditions from images.

## 🎯 Classification Categories

The model covers six classes:

1. Caterpillar and Semilooper Pest Attack
2. Frog Leaf Eye
3. Healthy
4. Mosaic
5. Rust
6. Spectoria Brown Spot

## 🛠️ Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Model Architecture:** EfficientFormer
- **Model Utilities:** timm
- **Image Processing:** OpenCV
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

## ⚙️ Project Workflow

1. **Dataset Preparation** — Organizing soybean leaf images for model development.
2. **Image Preprocessing** — Applying image transformations and preprocessing.
3. **Model Training** — Training deep learning models for six-class image classification.
4. **Model Evaluation** — Assessing model performance using accuracy, loss, and classification metrics.
5. **Visualization** — Analyzing model behavior through evaluation plots.
6. **Explainable AI** — Using Grad-CAM to inspect image regions that influence model predictions.

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **91.34%** |
| Training Accuracy | **80.62%** |
| Training Loss | **0.7472** |

*Validation accuracy and training accuracy refer to different data splits. These figures should be associated with the same experiment and configuration.*

## 📈 Evaluation and Visualization

The project includes tools for:

- Confusion matrix analysis
- Precision-recall curves
- ROC curves and AUC analysis
- Prediction confidence analysis
- Feature map visualization
- t-SNE visualization
- Model comparison

## 🔍 Explainable AI with Grad-CAM

Grad-CAM visualizations help identify image regions that influence the model's predictions. This supports qualitative inspection of model behavior and can help investigate whether the model focuses on relevant leaf features.

## 📁 Repository Structure

```text
soybean-disease-detection-portfolio/
├── README.md
├── requirements.txt
├── confusion_matrix.png
├── test_setup.py
└── src/
    ├── predict.py
    ├── evaluate.py
    ├── evaluate_baselines.py
    ├── compare_models.py
    ├── gradcam.py
    ├── feature_maps.py
    ├── confidence_curves.py
    ├── pr_curve.py
    ├── roc_curve.py
    ├── tsne_visualization.py
    ├── count_images.py
    ├── check_split.py
    ├── split_dataset.py
    ├── train.py
    ├── train_v2.py
    ├── test_model.py
    └── test_efficientformer.py
```

## 🎯 Project Objectives

- Develop a deep learning-based soybean leaf classification system.
- Explore EfficientFormer for image classification.
- Evaluate model performance using multiple metrics.
- Compare model behavior through visualizations.
- Investigate explainability techniques for image-based predictions.

## ⚠️ Limitations

This is an experimental machine learning project. Model performance may vary with dataset composition, image quality, and evaluation methodology. Predictions should not replace expert agricultural assessment.
