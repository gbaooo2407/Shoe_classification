# 🥿 XAI-KICKVISION: Explainable Multimodal AI for Shoe Recognition & Recommendation

A hybrid AI system that classifies and recommends shoes using both **images** and **product descriptions**, combining **ResNet18**, **Vision Transformer (ViT)**, and **DistilBERT**. This system leverages **multimodal learning** and **Grad-CAM++** to improve both **accuracy** and **interpretability**, optimized for **e-commerce applications**.

---

## 📌 Table of Contents

- [Features](#-features)
- [Project Goals](#-project-goals)
- [Architecture](#-architecture)
- [Techniques Used](#-techniques-used)
- [Technologies](#-technologies)
- [Dataset](#-dataset)
- [Results](#-results)
- [Grad-CAM++ Example](#-grad-cam-example)
- [How to Run](#-how-to-run)
- [Team](#-team)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🔍 Features

- ✅ Multimodal learning with **image + text**
- ✅ Integrated **ResNet18**, **ViT**, and **DistilBERT**
- ✅ Explainability via **Grad-CAM++**
- ✅ Supports **incremental learning (EWC)** and **model pruning**

---

## 🎯 Project Goals

- Recommend and classify shoes using product images and descriptions.
- Provide visual explanations for predictions (trustworthy AI).
- Enable efficient training and updating with new data (incremental learning).
- Optimize model for deployment (pruning).

---

## 🧠 Architecture


     +-----------+        +-------------+
     |  ResNet18 | -----> |             |
     +-----------+        |             |
                          |             |      
     +----------+         | Multimodal  |         +-----------------+
     | DistilBERT| -----> |   Fusion    | ----->  |  Classifier     |
     +----------+         |             |         +-----------------+
                          |             |
     +---------+          |             |
     |  ViT     | ------> |             |
     +---------+          +-------------+


- ResNet18 & ViT: extract visual features.
- DistilBERT: extract textual features.
- Fusion layer: combines modalities.
- Classifier: predicts product category or recommends similar items.

---

## 🧪 Techniques Used

- Multimodal Learning
- Grad-CAM++ (Explainability)
- EWC (Incremental Learning)
- Pruning (Model Compression)

---

## 🛠️ Technologies

- **PyTorch**
- **Torchvision**
- **HuggingFace Transformers**
- **Grad-CAM++**
- **NumPy**, **Pandas**, **Scikit-learn**

---

## 📁 Dataset

- Internal dataset of shoe images + product descriptions
- Labeled for classification and similarity recommendation
- Preprocessing:
  - Image: resized to 224x224, normalized
  - Text: tokenized via `DistilBERTTokenizer`

---

## 📈 Results

| Metric             | Value        |
|--------------------|--------------|
| Top-1 Accuracy     | 92.5%        |
| Model Size Reduced | 38% (via pruning) |
| Explainability     | Visualized with Grad-CAM++ |

---
