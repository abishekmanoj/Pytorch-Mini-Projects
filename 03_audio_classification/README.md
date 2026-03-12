# Music Genre Classification (PyTorch)

This mini project implements a **music genre classification model using PyTorch**.  
The model learns to classify audio tracks into different genres by converting audio signals into **Mel Spectrograms** and training a **Convolutional Neural Network (CNN)**.

This project is part of a **PyTorch Mini Projects series** designed to practice deep learning workflows across different data types.

---

# Dataset

Dataset used:

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

The dataset contains **1000 audio tracks** across **10 music genres**.

Each genre contains **100 audio samples**.

Genres included:

- Blues
- Classical
- Country
- Disco
- HipHop
- Jazz
- Metal
- Pop
- Reggae
- Rock

Each audio file is approximately **30 seconds long**.

---

# Project Pipeline

The workflow for this project:

```
Audio File (.wav)
        ↓
Load waveform
        ↓
Convert to Mel Spectrogram
        ↓
Log Scaling + Normalization
        ↓
CNN Model
        ↓
Genre Prediction
```

Audio is converted into a **Mel Spectrogram**, which is a time-frequency representation of sound and can be treated like an image.

---

# Data Preprocessing

Steps applied to audio data:

1. Load audio waveform using **torchaudio**
2. Convert stereo audio to **mono**
3. Resample audio to **22050 Hz**
4. Extract **3-second clips** from audio tracks
5. Convert waveform to **Mel Spectrogram**
6. Apply **log scaling**
7. Normalize spectrogram values
8. Pad or truncate spectrograms to fixed length

Mel Spectrogram parameters:

```
sample_rate = 22050
n_fft = 1024
hop_length = 512
n_mels = 64
```

---

# Model Architecture

A **Convolutional Neural Network (CNN)** is used for classification.

```
Input: Mel Spectrogram (1 × 64 × time)

Conv2D (1 → 16)
ReLU
MaxPool

Conv2D (16 → 32)
ReLU
MaxPool

Conv2D (32 → 64)
ReLU
MaxPool

Adaptive Average Pooling

Flatten

Fully Connected Layer

Output Layer (10 classes)
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

Training Epochs:

```
30
```

---

# Training Process

The model is trained using:

- Training dataset
- Validation dataset
- Test dataset

During training the following metrics are tracked:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Training curves are plotted to visualize learning behavior.

---

# Evaluation

Model performance is evaluated using:

- Test Accuracy
- Confusion Matrix
- Classification Report

The confusion matrix helps visualize how well the model predicts each music genre.

---

# Results

Final model performance (approximate):

```
Training Accuracy ≈ 47%
Validation Accuracy ≈ 47–52%
```

This is expected for a **small CNN trained on the GTZAN dataset** without heavy augmentation.

---

# Project Structure

```
03_audio_classification
│
├── data/
│   ├── blues
│   ├── classical
│   ├── country
│   ├── disco
│   ├── hiphop
│   ├── jazz
│   ├── metal
│   ├── pop
│   ├── reggae
│   └── rock
│
├── models/
│   └── audio_classifier.pth
│
├── train.ipynb
└── README.md
```

---

# Key Concepts Practiced

This project demonstrates:

- Audio data loading using **torchaudio**
- Feature extraction with **Mel Spectrograms**
- Audio preprocessing and normalization
- Building CNNs for audio tasks
- Training and validation loops in PyTorch
- Model evaluation and confusion matrix analysis
- Saving and loading trained models

---
