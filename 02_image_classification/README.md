# Animal Face Image Classification (PyTorch)

This mini project implements an **image classification model using PyTorch** to classify animal faces from images.

The goal of this project is to practice the **end-to-end PyTorch computer vision workflow**, including dataset preparation, image preprocessing, building a CNN model, training, validation, and evaluation.

---

## Dataset

Dataset used:  
https://www.kaggle.com/datasets/andrewmvd/animal-faces

The dataset contains images of animal faces organized into folders by class.

Example classes include:

- Cat
- Dog
- Wild animals

Each class contains multiple images of animal faces used for training and evaluation.

---

## Data Preprocessing

The dataset is converted into a dataframe containing:

- `image_path` → path to the image file
- `labels` → corresponding class label

Images are processed using **Torchvision transforms**:

```
Resize → 128x128
Convert to Tensor
Convert dtype to float
```

The dataset is then split into:

- **Train set** (70%)
- **Validation set** (15%)
- **Test set** (15%)

A custom PyTorch `Dataset` class is used to load images and labels.

---

## Model Architecture

A simple **Convolutional Neural Network (CNN)** is used.

```
Input Image (3 x 128 x 128)
        ↓
Conv2D (3 → 16)
ReLU
MaxPool
        ↓
Conv2D (16 → 32)
ReLU
MaxPool
        ↓
Conv2D (32 → 64)
ReLU
MaxPool
        ↓
Flatten
        ↓
Fully Connected Layer
        ↓
Output Layer (number of classes)
```

Loss Function:

```
CrossEntropyLoss
```

Optimizer:

```
Adam
```

Batch Size:

```
16
```

---

## Training Pipeline

The training process follows a standard PyTorch workflow:

1. Load dataset paths and labels
2. Encode labels using `LabelEncoder`
3. Apply image transformations
4. Create custom `Dataset`
5. Use `DataLoader` for batching
6. Train the CNN model
7. Track **training and validation loss**
8. Track **training and validation accuracy**
9. Evaluate on the test dataset
10. Save the trained model

---

## Evaluation

The model performance is evaluated using:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Confusion Matrix

Training curves are plotted to visualize model learning.

---

## Project Structure

```
.
├── data/
│   └── train
|   └── val
│
├── models/
│   └── image_classifier.pth
│
├── train.ipynb
├── README.md
└── .gitignore
```

- **data/** → dataset images
- **models/** → saved trained model
- **train.ipynb** → full training pipeline
- **README.md** → project documentation

---