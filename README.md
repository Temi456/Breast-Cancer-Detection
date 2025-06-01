# Breast-Cancer-Detection
Python scripts Spyder IDE
# Breast Cancer Detection using Convolutional Neural Networks (CNN)

This project applies deep learning (CNNs with transfer learning) to classify mammogram images into three categories: **benign**, **normal**, and **malignant**. The model is built using PyTorch and is trained on the MIAS dataset.

---

## Objective

To support early detection of breast cancer by automating image classification from mammogram scans, with potential to assist radiologists in clinical diagnosis.

---

## 🗂 Dataset

- **Name:** MIAS (Mammographic Image Analysis Society)
- **Type:** Grayscale mammogram images
- **Format:** `.pgm` images with associated labels in a CSV file
- **Classes:**
  - `0`: Benign
  - `1`: Normal
  - `2`: Malignant

---

## Model Architecture

- **Backbone:** Pretrained ResNet18
- **Modifications:**
  - First layer adapted to accept grayscale (converted to RGB)
  - Final FC layer replaced with custom classification head for 3 classes
  - Dropout added for regularization
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss with optional class weighting
- **Learning Rate Scheduler:** ReduceLROnPlateau

---

## 🛠 Features

- Custom PyTorch `Dataset` class with class-aware augmentations
- Weighted sampling to balance class distribution
- Modular training loop with validation monitoring
- Visualization of accuracy/loss curves and confusion matrix

---

## 📈 Results

| Metric      | Score     |
|-------------|-----------|
| Accuracy    | **82%**   |
| Malignant Recall | **95%** |
| F1-Score (macro avg) | **81%** |

> Model demonstrated high precision and recall for malignant cases, reducing risk of false negatives.

---

## Lessons Learned

- Class imbalance must be handled early using augmentation and sampling.
- Transfer learning is effective for small medical datasets.
- Clear code structure and documentation improve reproducibility and collaboration.

---

## Future Work

- Apply model to other imaging modalities (CT, MRI)
- Integrate into clinical diagnostic workflows
- Expand to multi-modal models using clinical metadata

---

## 🧪 Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas, numpy, matplotlib, seaborn, scikit-learn, PIL

```bash
pip install -r requirements.txt
