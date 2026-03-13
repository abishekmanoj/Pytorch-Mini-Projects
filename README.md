# PyTorch Mini Projects

This repository contains a series of small projects built to practice
deep learning workflows using PyTorch.

Each project focuses on a different type of data and model.

## Projects

### 1. Tabular Data Classification
Location: `01_rice_type_classification`

Dataset:
https://www.kaggle.com/datasets/mssmartypants/rice-type-classification

Binary classification using the Rice Type dataset.

Topics covered:
- Data preprocessing
- PyTorch Dataset and DataLoader
- Feedforward neural networks
- Training loops
- Evaluation and model saving

---

### 2. Image Classification
Location: `02_image_classification`

CNN model trained on the Animal Faces dataset.

Dataset:
https://www.kaggle.com/datasets/andrewmvd/animal-faces

Topics covered:
- Image preprocessing
- Torchvision transforms
- Custom image dataset
- Convolutional neural networks
- Training / validation loops
- Confusion matrix evaluation

---

### 3. Audio Classification
Location: `03_audio_classification`

CNN model trained on Mel Spectrogram representations of audio signals
to classify music genres.

Dataset:  
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Topics covered:
- Audio loading with **torchaudio**
- Waveform preprocessing
- Mel Spectrogram feature extraction
- Log scaling and normalization
- CNN-based audio classification
- Training and validation loops
- Model evaluation using confusion matrix

---

### 4. Text Classification
Location: `04_text_classification`

Sarcasm detection using a **BERT-based transformer model**.

Dataset:  
https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

Topics covered:

- NLP preprocessing
- Tokenization using HuggingFace Transformers
- BERT embeddings
- Transformer-based text classification
- PyTorch Dataset and DataLoader for text
- Training and evaluation of transformer models

---

# Future Improvements

Possible extensions for this repository:

- Transfer Learning with ResNet / EfficientNet
- Vision Transformers (ViT)
- Advanced NLP models
- Model deployment with FastAPI
- Experiment tracking with MLflow or Weights & Biases

---
