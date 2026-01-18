# Pneumonia X-Ray Detector 🩻🤖

An AI project that classifies chest X-ray images into **Normal** or **Pneumonia** using **EfficientNet-B4** with **transfer learning** in **PyTorch**.

## 📌 Overview
Pneumonia is a serious lung infection that can be detected using chest X-ray images, but manual diagnosis requires time and expert knowledge.  
This project builds an automated system to assist in identifying pneumonia cases quickly and consistently.

## 📂 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:
- Training: 5,216 images (imbalanced)
- Validation: 16 images
- Test: 624 images

Classes:
- Normal
- Pneumonia

Dataset link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## ⚙️ Preprocessing
To match ImageNet-pretrained model requirements, the dataset is processed using `torchvision.transforms`:
- Resize and center-crop to **224×224**
- Convert to grayscale and expand to **3 channels**
- Normalize using ImageNet mean & std
- Training augmentation: random rotation (-20° to +20°)

## 🧠 Model
We use **EfficientNet-B4** pretrained on ImageNet:
- Backbone extracts features from X-ray images
- Classification head is replaced with a custom layer for binary classification
- Output is passed through a sigmoid for prediction

## 🏋️ Training
- Framework: **PyTorch**
- Optimizer: **AdamW**
- Learning Rate: **0.001**
- Epochs: **20**

## 📊 Evaluation Results
The model is evaluated on a hidden test set of 624 images.

✅ Final Performance:
- **Accuracy:** 94.71%
- **Test Loss:** 0.2187

Key Metrics:
- Precision: 0.95
- Recall: 0.96
- F1-score: 0.96

Confusion Matrix:
- Normal → Normal: 215
- Normal → Pneumonia: 19
- Pneumonia → Pneumonia: 376
- Pneumonia → Normal: 14

## 🚀 How to Run
1. Clone the repo:
```bash
git clone https://github.com/your-username/pneumonia-xray-detector.git
cd pneumonia-xray-detector
