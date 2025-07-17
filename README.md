

Link to notebook: https://colab.research.google.com/drive/1NVzNyw_YhNmemHSyJSiKQVESM-YRQkop?usp=sharing

# 🐾 Pet Disease Contrastive Learning Classification

A deep‐learning project that leverages **contrastive learning** (triplet sampling) and transfer learning to build a robust pet disease classifier. The model learns to pull together embeddings of images from the same disease class and push apart embeddings of different classes, improving inter-class discrimination in veterinary diagnosis tasks.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Data Loading & Triplet Sampling](#data-loading--triplet-sampling)  
4. [Model Architectures](#model-architectures)  
5. [Data Analysis](#data-analysis)
6. [Learning Curves](#learning-curves)
7. [Evaluation & Visualization](#evaluation--visualization)  

---

## Project Overview

This repository implements a **contrastive learning** pipeline for multi‐class pet disease classification. By sampling **triplets**—a **query** (anchor) image, a **positive** image (same class), and a **negative** image (different class)—the model learns an embedding space where diseases cluster tightly and are well separated from other conditions.

Key features:
- 📥 Automated data download & organization  
- 🔄 Triplet dataset sampling (`CustomDataset`)  
- 🔄 Data augmentations & normalization  
- ⚙️ Transfer learning with **ReXNet-150** backbone  
- 🔍 Interpretability via **Grad-CAM**  
- 📊 Confusion matrix & learning curves  

---

##  Dataset

**Source**: Kaggle “Pet Disease Images”  
- Organized on disk as:
```datasets/pet_disease/data/
├── ringworm_in_cat/
│   ├── img1.jpg
│   └── …
├── dental_disease_in_dog/
│   └── …
└── …
```

- 7+ disease classes covering both dogs and cats.

---

## Data Loading & Triplet Sampling

- **`CustomDataset`**  
- Recursively collects all image paths.  
- Builds `cls_names` ↔ integer mappings.  
- Implements `__getitem__` to return a dict:
  ```python
  {
    "qry_im":   Tensor,  # anchor image
    "pos_im":   Tensor,  # same‐class positive
    "neg_im":   Tensor,  # different‐class negative
    "qry_gt":   int,     # anchor label
    "neg_gt":   int      # negative label
  }
  ```
- **Transforms**  
- Resize to 224×224, ToTensor & Normalize using ImageNet statistics.
- **`get_dls()`**  
- Splits dataset into train/val/test (default 90/5/5).  
- Returns PyTorch `DataLoader`s with triplet batches.

---

## Model Architectures

 **Embedding Network**  
 - Base: Pretrained **ReXNet-150**  
 - Customized classification head → embedding vector.


---

## Data Analysis

- **Class Imbalance Analysis**
<img width="901" height="664" alt="image" src="https://github.com/user-attachments/assets/898bca16-e8bb-4168-829f-16e521472117" />

- **Sample Triplet Images**:
  <img width="1696" height="1603" alt="image" src="https://github.com/user-attachments/assets/b088a3d0-2d79-47fb-a080-bf18cbcd07f8" />

### Learning Curves

| Loss Curve | Accuracy & F1 Score | Sensitivity & Specificity |
|------------|---------------------|----------------------------|
 | <img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/68a59d38-ddbe-4e46-9bc2-26b8de663fd2" /> | <img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/8a409016-a63c-4bf7-8e2e-33a5cd1349b5" /> | <img width="863" height="470" alt="image" src="https://github.com/user-attachments/assets/3ce6a56b-b99e-45c2-b001-1595769a2221" /> |

---



## Evaluation & Visualization


- **Confusion Matrix**: Multiclass performance overview.
<img width="1623" height="1041" alt="image" src="https://github.com/user-attachments/assets/75a5264a-6877-4111-8104-2d2b697c3740" />

- **Grad-CAM**: Class activation maps highlighting regions that influenced predictions.
<img width="1785" height="1021" alt="image" src="https://github.com/user-attachments/assets/b7e4519d-336e-4ac6-895e-5daa2dc918fb" />



