

Link to notebbok: https://colab.research.google.com/drive/1NVzNyw_YhNmemHSyJSiKQVESM-YRQkop?usp=sharing

# 🐾 Pet Disease Contrastive Learning Classification

A deep‐learning project that leverages **contrastive learning** (triplet sampling) and transfer learning to build a robust pet disease classifier. The model learns to pull together embeddings of images from the same disease class and push apart embeddings of different classes, improving inter-class discrimination in veterinary diagnosis tasks.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Data Loading & Triplet Sampling](#data-loading--triplet-sampling)  
4. [Model Architectures](#model-architectures)  
5. [Training & Loss](#training--loss)  
6. [Evaluation & Visualization](#evaluation--visualization)  
7. [Usage](#usage)  
8. [Dependencies](#dependencies)  
9. [Project Structure](#project-structure)  
10. [Future Improvements](#future-improvements)  
11. [License](#license)  

---

## 🚀 Project Overview

This repository implements a **contrastive learning** pipeline for multi‐class pet disease classification. By sampling **triplets**—a **query** (anchor) image, a **positive** image (same class), and a **negative** image (different class)—the model learns an embedding space where diseases cluster tightly and are well separated from other conditions.

Key features:
- 📥 Automated data download & organization  
- 🔄 Triplet dataset sampling (`CustomDataset`)  
- 🔄 Data augmentations & normalization  
- ⚙️ Transfer learning with **ReXNet-150** backbone  
- 🔍 Interpretability via **Grad-CAM**  
- 📊 Confusion matrix & learning curves  

---

## 📂 Dataset

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

## 🔄 Data Loading & Triplet Sampling

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

## 🏗️ Model Architectures

1. **Embedding Network**  
 - Base: Pretrained **ReXNet-150**  
 - Customized classification head → embedding vector.

2. **Projection Head (optional)**  
 - MLP mapping embeddings to contrastive space (if using contrastive loss like NT-Xent).

---

## 🧪 Training & Loss

- **Contrastive Loss** (e.g., Triplet Margin Loss or NT-Xent Loss)  
- **Optimizer**: AdamW (or SGD + cosine LR)  
- **Metrics**:  
- Top-1 classification accuracy on test set  
- Triplet margin validation loss  

---

## 📊 Evaluation & Visualization

- **Learning Curves**: Train/val loss & accuracy plots.  
- **Confusion Matrix**: Multiclass performance overview.  
- **Grad-CAM**: Class activation maps highlighting regions that influenced predictions (see `gradcam/` folder).

---

## ⚙️ Usage

```bash
# 1. Clone and enter project
git clone https://github.com/your-username/pet-disease-contrastive.git
cd pet-disease-contrastive

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

# 4. Download & extract dataset
python scripts/download_data.py --dataset pet-disease-images

# 5. Train
python train.py \
--data_root ./datasets/pet_disease/data \
--backbone rexnet_150 \
--epochs 30 \
--batch_size 16 \
--lr 1e-4 \
--contrastive_loss triplet

# 6. Evaluate
python evaluate.py --checkpoint runs/exp1/model_best.pth

# 7. Visualize GradCAM
python visualize_gradcam.py --image_path assets/cat_ringworm.jpg \
--checkpoint runs/exp1/model_best.pth
