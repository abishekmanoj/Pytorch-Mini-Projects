# Sarcasm Detection using BERT (PyTorch)

This mini project implements a **text classification model using PyTorch and a pretrained BERT encoder** to detect sarcasm in news headlines.

The project demonstrates how to fine-tune transformer-based language models for **binary NLP classification tasks**.

This is part of a **PyTorch Mini Projects series** covering different deep learning data types.

---

# Dataset

Dataset used:

https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

The dataset contains **news headlines labeled as sarcastic or not sarcastic**.

Columns in the dataset:

- `headline` → News headline text
- `is_sarcastic` → Target label (0 or 1)
- `article_link` → Original article source (removed during preprocessing)

Label meaning:

```
0 → Not sarcastic
1 → Sarcastic
```

---

# Project Pipeline

The workflow for this project:

```
Text Headline
      ↓
Tokenizer (BERT Tokenizer)
      ↓
Input IDs + Attention Mask
      ↓
BERT Encoder
      ↓
CLS Token Representation
      ↓
Fully Connected Classifier
      ↓
Sarcasm Prediction
```

---

# Data Preprocessing

Steps applied to the dataset:

1. Load dataset from JSON
2. Remove missing values
3. Remove duplicate rows
4. Drop unused column (`article_link`)
5. Split dataset into:
   - Training set (70%)
   - Validation set (15%)
   - Test set (15%)
6. Tokenize headlines using **BERT tokenizer**
7. Convert text into:
   - `input_ids`
   - `attention_mask`
8. Pad sequences to a fixed maximum length

Tokenizer used:

```
bert-base-uncased
```

Maximum sequence length:

```
64 tokens
```

---

# Model Architecture

The model uses a **pretrained BERT encoder** with a custom classification head.

```
Input Headline
      ↓
BERT Tokenizer
      ↓
BERT Encoder (768 hidden size)
      ↓
CLS Token Embedding
      ↓
Linear Layer (768 → 256)
      ↓
ReLU
      ↓
Dropout
      ↓
Linear Layer (256 → 2)
      ↓
Output Classes
```

Output classes:

```
0 → Not Sarcastic
1 → Sarcastic
```

---

# Training Configuration

Loss Function:

```
CrossEntropyLoss
```

Optimizer:

```
AdamW
```

Learning Rate:

```
2e-5
```

Batch Size:

```
16
```

Training Epochs:

```
3
```

---

# Evaluation

The model is evaluated using:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Confusion Matrix
- Classification Report

Training curves are plotted to visualize learning progress.

---

# Expected Performance

With BERT fine-tuning, the model typically achieves:

```
Validation Accuracy ≈ 88–92%
```

on this dataset.

---

# Project Structure

```
04_text_classification
│
├── data/
│   └── dataset.json
│
├── models/
│   └── sarcasm_classifier.pth
│
├── train.ipynb
└── README.md
```

Explanation:

- **data/** → dataset files
- **models/** → saved trained model
- **train.ipynb** → full training pipeline
- **README.md** → project documentation

---

# Key Concepts Practiced

This project demonstrates:

- Transformer-based NLP models
- Text tokenization using HuggingFace Transformers
- BERT embeddings
- Custom PyTorch Dataset for text
- Training and validation loops
- Binary classification with deep learning
- Model evaluation with confusion matrix
- Saving and loading trained models

---

# PyTorch Mini Projects Series

This repository contains multiple deep learning mini-projects:

1. Tabular Data Classification
2. Image Classification
3. Audio Classification
4. Text Classification

The goal is to build practical experience with **PyTorch across multiple data modalities**.

---