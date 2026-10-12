# Pneumonia X-Ray Detector

Binary classification of chest X-rays as **Normal** or **Pneumonia** by fine-tuning an ImageNet-pretrained EfficientNet-B4 in PyTorch.

## Problem

Reading chest X-rays for pneumonia takes time and expert knowledge. This project tests how well a transfer-learned CNN can separate normal from pneumonia X-rays on a public dataset.

## Approach

**Data:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle, using its own `train/`, `val/` and `test/` folders. The training set is imbalanced toward Pneumonia.

**Preprocessing and augmentation** (`torchvision.transforms`)
- Train: resize to 256x256, random rotation between -20 and +20 degrees, center crop to 224x224
- Validation / test: resize to 224x224
- All splits: grayscale expanded to 3 channels, then ImageNet mean / std normalization

**Model**
- `torchvision` EfficientNet-B4 with ImageNet weights
- Final classifier layer replaced by a single-logit linear layer
- Prediction: sigmoid of the logit, threshold 0.5

**Training:** AdamW (learning rate 1e-3), `BCEWithLogitsLoss`, batch size 32, all layers trainable.

## Results

**Evaluation setup:** the Kaggle test folder (624 images: 234 Normal, 390 Pneumonia), not used for training. Precision, recall and F1 treat Pneumonia as the positive class.

| Metric | Test set |
|---|---|
| Accuracy | 94.71% |
| Precision | 0.952 |
| Recall | 0.964 |
| F1 | 0.958 |

Confusion matrix (rows = true label, columns = predicted):

| | Pred. Normal | Pred. Pneumonia |
|---|---|---|
| **Normal** | 215 | 19 |
| **Pneumonia** | 14 | 376 |

**Caveats**
- The saved notebook output comes from a **1-epoch** training run, even though the code is now set to `num_epochs = 20`. These numbers have not been reproduced with the current setting.
- The validation set is only **16 images**, so validation accuracy is not a reliable signal for model selection.

## Tech Stack

Python, PyTorch, torchvision, scikit-learn (metrics), NumPy, pandas, matplotlib, seaborn, OpenCV, Jupyter.

## How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/Faris-z/pneumonia-xray-detector.git
   cd pneumonia-xray-detector
   ```

2. Install dependencies (there is no requirements file yet):

   ```bash
   pip install torch torchvision scikit-learn numpy pandas matplotlib seaborn opencv-python tqdm jupyter
   ```

3. Download the [Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it next to the notebook as:

   ```
   chest_xray/
   ├── train/{NORMAL,PNEUMONIA}/
   ├── val/{NORMAL,PNEUMONIA}/
   └── test/{NORMAL,PNEUMONIA}/
   ```

4. Open `sourceCode.ipynb` and run all cells. A CUDA GPU is used if available.

## Limitations / Next Steps

- Re-run the full 20-epoch training and save the learning curves and the final test output.
- Build a proper validation set (for example a stratified split of the training data) instead of the 16-image folder.
- Handle the class imbalance (class weighting or a tuned decision threshold) and report ROC-AUC.
- The trained weights are not saved. Next: save a checkpoint and add a small inference script.
- Add a `requirements.txt` with pinned versions.
- Replace the deprecated `pretrained=True` argument with `weights=EfficientNet_B4_Weights.IMAGENET1K_V1`.
- Research project only, not a medical device.
