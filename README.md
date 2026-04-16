<h1 align="center"> Sign Language Detection using ANN & CNN</h1>

<p align="center">
  <strong>A comparative study of Artificial Neural Networks (ANN) vs Convolutional Neural Networks (CNN) for sign language recognition on the Sign MNIST dataset - part of a final-year B.Tech project.</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
  <a href="#"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"></a>
  <a href="#"><img src="https://img.shields.io/badge/Final%20Year-Project-purple?style=for-the-badge" alt="Final Year"></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-ann-vs-cnn">ANN vs CNN</a> •
  <a href="#-model-architectures">Architectures</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-comparison">Comparison</a>
</p>

---

## 📖 Overview

This project compares two deep learning approaches for **American Sign Language (ASL) letter recognition**:

1. **ANN (Artificial Neural Network)** - A fully-connected feedforward network that treats images as flat pixel vectors
2. **CNN (Convolutional Neural Network)** - A convolutional architecture that preserves spatial structure and learns hierarchical visual features

Both models are trained on the **Sign MNIST dataset** to classify hand gesture images into 24 ASL letter classes (A–Y, excluding J and Z which require motion).

> **Companion Repository:** This is the baseline comparison. The object detection approach using YOLOv5 can be found in [Sign-Language-Detection-Using-YOLO-V5](https://github.com/zishnusarker/Sign-Language-Detection-Using-YOLO-V5).

---

## 📊 Dataset

### Sign MNIST

The **Sign Language MNIST** dataset is a drop-in replacement for the classic MNIST, formatted identically but with hand gesture images instead of handwritten digits.

| Property | Value |
|----------|-------|
| **Image Size** | 28 × 28 pixels (grayscale) |
| **Training Samples** | 27,455 images |
| **Test Samples** | 7,172 images |
| **Classes** | 24 letters (A–Y, excluding J & Z) |
| **Format** | CSV (label + 784 pixel values per row) |
| **Source** | [Kaggle Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) |

### Why exclude J and Z?

The letters **J** and **Z** require **motion** to perform - J involves a downward arc and Z traces a zigzag. Since single static images can't capture motion, they're excluded from this image classification task.

### Gesture Mapping

```
A=0  B=1  C=2  D=3  E=4  F=5  G=6  H=7  I=8
K=10 L=11 M=12 N=13 O=14 P=15 Q=16 R=17 S=18
T=19 U=20 V=21 W=22 X=23 Y=24

(J=9 and Z=25 are excluded - they require motion)
```

---

## 🧠 ANN vs CNN

| Aspect | ANN (Fully Connected) | CNN (Convolutional) |
|--------|----------------------|---------------------|
| **Input handling** | Flattens 28×28 → 784-pixel vector | Keeps 28×28×1 spatial structure |
| **Feature learning** | Learns from raw pixel values | Learns spatial features (edges, textures, shapes) |
| **Parameters** | Many (every pixel to every neuron) | Fewer (shared convolutional filters) |
| **Translation invariance** | None - position matters | Built-in - same gesture anywhere |
| **Best for** | Simple patterns, tabular data | Images, spatial data |
| **Expected accuracy** | ~70–80% | ~90–95%+ |

---

## 🏗 Model Architectures

### ANN Architecture (ANN.ipynb)

```
Input (784 pixels - flattened 28×28)
        │
   Dense(128/256, activation='relu')     ← Hidden Layer 1
        │
   Dense(128, activation='relu')         ← Hidden Layer 2
        │
   Dense(24, activation='softmax')       ← Output (24 ASL classes)
```

The ANN treats each image as a **flat vector of 784 numbers**. It has no concept of pixel neighborhoods - pixel (1,1) and pixel (28,28) are treated equally.

### CNN Architecture (CNN.ipynb)

```
Input (28 × 28 × 1 grayscale image)
        │
   Conv2D(32, 3×3, activation='relu')   ← Edge/texture detection
   MaxPooling2D(2×2)                     ← Spatial downsampling
        │
   Conv2D(64, 3×3, activation='relu')   ← Shape detection
   MaxPooling2D(2×2)                     ← Further downsampling
        │
   Flatten()                             ← Convert to 1D
        │
   Dense(128, activation='relu')         ← Classification
        │
   Dense(24, activation='softmax')       ← Output (24 ASL classes)
```

The CNN preserves **2D spatial structure**. Convolutional layers learn local patterns (edges → textures → hand shapes), and pooling layers progressively reduce spatial dimensions.

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.7+ | Core implementation |
| **Deep Learning** | TensorFlow / Keras | Model building and training |
| **Data** | NumPy, Pandas | Data loading and preprocessing |
| **Visualization** | Matplotlib | Training curves and results |
| **Notebook** | Jupyter | Interactive development |
| **Dataset** | Sign MNIST (CSV) | 28×28 hand gesture images |

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/zishnusarker/Sign-language-Detection-Using-ANN-CNN.git
cd Sign-language-Detection-Using-ANN-CNN

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install tensorflow numpy pandas matplotlib jupyter scikit-learn
```

---

## 📋 Usage

### Run the ANN Model

```bash
jupyter notebook ANN.ipynb
```

Run all cells to: load data → flatten images to 784-d vectors → build dense network → train → evaluate

### Run the CNN Model

```bash
jupyter notebook CNN.ipynb
```

Run all cells to: load data → reshape to 28×28×1 → build Conv2D network → train → evaluate and compare

---

## 📁 Project Structure

```
Sign-language-Detection-Using-ANN-CNN/
├── README.md                    # Project documentation
├── ANN.ipynb                    # ANN implementation notebook
├── CNN.ipynb                    # CNN implementation notebook
├── sign_mnist_train.csv         # Training data (27,455 samples)
├── sign_mnist_test.csv          # Test data (7,172 samples)
└── .gitattributes               # Git config
```

---

## 📊 Full Comparison: ANN vs CNN vs YOLOv5

This repo is part of a **three-way comparison** conducted as a 7th semester final-year project:

| Model | Repository | Approach | Best For |
|-------|-----------|----------|----------|
| **ANN** | This repo | Flat pixel classification | Baseline benchmark |
| **CNN** | This repo | Spatial feature learning | Static image classification |
| **YOLOv5** | [Sign-Language-Detection-Using-YOLO-V5](https://github.com/zishnusarker/Sign-Language-Detection-Using-YOLO-V5) | Object detection + localization | Real-time webcam detection |

### Key Findings

- **ANN** establishes a baseline - raw pixel classification has fundamental limitations for image tasks
- **CNN** proves the power of spatial feature learning - convolutional layers extract patterns dense layers cannot
- **YOLOv5** adds real-time detection, bounding boxes, and multi-sign recognition

---

## 🎓 Key Concepts Demonstrated

<details>
<summary><strong>Why do CNNs outperform ANNs on images?</strong></summary>

ANNs treat every pixel independently - pixel (1,1) has no relationship to pixel (1,2). But images have **spatial structure**: neighboring pixels form edges, edges form textures, textures form shapes. CNNs exploit this with local receptive fields, weight sharing, translation invariance, and hierarchical feature learning.

</details>

<details>
<summary><strong>What is MaxPooling and why is it used?</strong></summary>

MaxPooling reduces spatial dimensions by taking the maximum value in each patch (e.g., 2×2). It reduces computational cost, provides slight translation invariance, and forces the network to learn increasingly abstract features.

</details>

<details>
<summary><strong>Why Softmax for 24-class output?</strong></summary>

Softmax converts raw outputs into a probability distribution across all 24 classes where probabilities sum to 1.0. For multi-class classification, it gives interpretable results like "85% A, 10% B, 5% C."

</details>

<details>
<summary><strong>Why exclude J and Z from the dataset?</strong></summary>

J and Z are **dynamic signs** requiring motion (J traces a curve, Z traces a zigzag). Static images can't capture this. Recognizing them would require temporal models (LSTMs or video-based CNNs).

</details>

---

## 🔮 Future Improvements

- Add confusion matrix and classification report for both models
- Implement data augmentation to improve generalization
- Try deeper architectures (ResNet, VGG via transfer learning)
- Add dropout regularization to reduce overfitting
- Deploy as a web app using Streamlit or Flask
- Extend to video-based recognition to include J and Z
- Add real-time webcam prediction
- Visualize learned convolutional filters

---

## 📄 License

This project is available as open source. <strong>Understanding why architecture matters in deep learning </strong>

---
