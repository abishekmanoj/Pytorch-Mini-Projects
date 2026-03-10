# PROJECT 01 - Rice Type Classification (PyTorch)

This mini project implements a **binary classification model using PyTorch** to classify rice grains based on morphological features from a tabular dataset.

The goal of this project is to practice the **end-to-end PyTorch workflow for tabular data**, including data preprocessing, dataset creation, model training, evaluation, and model saving.

---

## Dataset

Dataset: **Rice Type Classification**

Download Link: ``` https://www.kaggle.com/datasets/mssmartypants/rice-type-classification ```

The dataset contains **11 columns**:

- 10 numerical features describing rice grain shape and geometry
- 1 target column (`Class`) representing the rice type

Example features include:

- Area
- MajorAxisLength
- MinorAxisLength
- Eccentricity
- ConvexArea
- Perimeter
- Roundness
- AspectRatio

The task is to classify the rice grain into **two classes (0 or 1)**.

---

## Model Architecture

A simple **Fully Connected Neural Network** is used.

```
Input Layer (10 features)
        ↓
Linear Layer (10 → 10)
        ↓
ReLU Activation
        ↓
Linear Layer (10 → 1)
        ↓
Sigmoid (via BCEWithLogitsLoss)
```

Loss Function:

```
BCEWithLogitsLoss
```

Optimizer:

```
Adam
```

Batch Size:

```
8
```

---

## Training Pipeline

The training pipeline follows the standard PyTorch workflow:

1. Load dataset using **pandas**
2. Split data into **train / validation / test**
3. Convert data to **PyTorch tensors**
4. Create custom **Dataset and DataLoader**
5. Train the neural network
6. Track **loss and accuracy**
7. Evaluate the model on validation and test sets
8. Save the trained model

---

## Project Structure

```
.
├── data/
│   └── riceClassification.csv
│
├── models/
│   └── rice_classifier.pth
│
└── train.ipynb
```

- **data/** → dataset used for training
- **models/** → saved trained model
- **train.ipynb** → training, evaluation, and plotting code

---

## Results

The model is trained for **10 epochs** and evaluated using:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Training curves are plotted to visualize the learning process.

---

## Purpose of This Project

This project is part of a **series of PyTorch mini projects** focused on learning deep learning fundamentals across different data modalities.

Projects in the series include:

- Tabular Data Classification (this project)
- Image Classification
- Transfer Learning for Image Classification
- Audio Classification
- Text Classification

The goal is to build a **strong practical foundation in PyTorch model development and experimentation**.

---