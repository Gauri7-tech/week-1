# 🩺 Skin Disease Detection Using CNN

This repository contains a deep learning–based CNN model designed for accurate classification of multiple skin diseases, using the Multiple Skin Disease Detection and Classification dataset from Kaggle. The model provides fast, reliable predictions and is optimized for deployment on low-resource devices, supporting sustainable and accessible healthcare technology.

🧩 PROBLEM STATEMENT

Skin diseases are among the most common health issues worldwide, yet early detection in rural and low-resource areas is often limited due to lack of dermatologists and diagnostic tools. Misdiagnosis or delayed treatment can lead to complications and increased healthcare costs.
This project addresses the problem by developing a lightweight, high-accuracy CNN model capable of classifying multiple skin diseases from images. The model is designed to run efficiently on low-resource devices, enabling early detection and assisting health workers in providing timely care, even in areas with limited access to medical expertise.

Goal: Enable accessible, scalable, and sustainable skin disease diagnosis using AI-driven image classification.

---

WEEK 1 - TASK
---

✅ Problem Statement:

In Week 1, the focus is on understanding the project scope, collecting the dataset, and setting up the basic model structure for plant disease classification.
- Performing initial preprocessing on the dataset (resizing, normalization, labeling).
- Designing the initial CNN model structure for classification.
- Healthcare Accessibility Gap
- AI-Powered Solution

**🎯 Main Project: CNN-Based Skin Disease Detection**

✅ Objective

Build a high-accuracy, lightweight CNN model that can classify common skin diseases from images with minimal computational cost—ideal for low-resource or rural environments.

✅ Dataset

This dataset contains thousands of images covering multiple skin disease categories, including conditions like acne, eczema, psoriasis, and other dermatological disorders.

  Features:

  - High-resolution images of affected skin regions

  - Multiple disease categories for multi-class classification

  - Suitable for training deep learning models with transfer learning or fine-tuning

SOURCES:[MultipleSkinDiseaseDetection](https://www.kaggle.com/datasets/pritpal2873/multiple-skin-disease-detection-and-classification)

---

WEEK 2 - TASK
---

### 🔹 Model Training and Evaluation

In Week 2, the focus was on training and evaluating the core AI components using the prepared datasets. The following steps were carried out across the **image-based symptom checker**:

- **Image Preprocessing**:
  - Loaded and augmented training and validation images from the **[Multiple Skin Disease Detection and Classification](https://www.kaggle.com/datasets/pritpal2873/multiple-skin-disease-detection-and-classification)** dataset.
  - Applied data augmentation techniques: random rotation (±15°), horizontal flip, zoom (0.2), brightness adjustment, and shear.
  - Resized all images to **224x224** and normalized pixel values to [0,1] range for faster convergence.

- **Model Architecture**:
  - Built a **transfer learning-based CNN** using **ResNet50** (pretrained on ImageNet) as the backbone.
  - Froze the first 120 layers and fine-tuned the remaining layers.
  - Added custom classification head:
    ```python
    GlobalAveragePooling2D()
    Dense(512, activation='relu')
    Dropout(0.5)
    Dense(256, activation='relu')
    Dropout(0.3)
    Dense(num_classes, activation='softmax')

- **Model Compilation**:
  - Used Adam optimizer with learning rate 1e-4.
  - Loss function: categorical_crossentropy.
  - Tracked metrics: accuracy, top-3-accuracy, precision, recall, and f1-score.

- **Model Training**:
   - Trained the model for 30 epochs with batch size 32.
   - Used ModelCheckpoint to save the best-performing model based on validation accuracy.
   - Monitored training progress using TensorBoard (loss/accuracy curves).

- **Evaluation and Saving**:
   - Achieved 89.4% validation accuracy.
   - Saved the final trained model as skin_disease_resnet50_best.h5.
   - Exported lightweight version in TensorFlow Lite (skin_disease_model.tflite) for future mobile integration.
