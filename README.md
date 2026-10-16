# 🔥 Wildfire Prediction Using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-CNN-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Metrics-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end deep learning framework for early wildfire detection and image classification using custom Convolutional Neural Networks (CNNs).**

</div>

---

## 📖 Overview

Wildfires cause catastrophic devastation to ecosystems, forestry, wildlife, and surrounding communities. Detecting fires in their nascent stages is essential for rapid containment and minimizing loss.

This repository implements a **binary image classification pipeline** using a custom **Convolutional Neural Network (CNN)** developed with **TensorFlow / Keras**. The project spans the entire machine learning lifecycle:
- Automated dataset loading & labeling from structured directories
- Exploratory Data Analysis (EDA) & multi-class sample image visualization
- Data augmentation using **Albumentations** (rotations, flips)
- Scalable batch data feeding with **Keras `ImageDataGenerator`**
- Custom deep CNN model architecture with **L2 Regularization** and **Dropout**
- Comprehensive post-training evaluation: **Confusion Matrix**, **Classification Report (Precision, Recall, F1)**, **Training History Plots**, and **Qualitative Sample Predictions**

---

## 🎯 Project Objectives

- 🌲 **Classify Images:** Accurately classify landscape / aerial imagery into `wildfire` and `nowildfire`.
- 🔄 **Data Augmentation:** Apply rotational and spatial transformations to boost generalizability and prevent overfitting.
- 🧠 **Custom CNN Architecture:** Build and optimize a multi-layer deep network from scratch with regularization techniques.
- 📊 **Exhaustive Evaluation:** Go beyond raw training accuracy by inspecting AUC, Confusion Matrices, Precision, Recall, and F1-Scores on unseen test data.
- 💾 **Automated Checkpointing:** Retain the optimal model weights using `ModelCheckpoint` and `EarlyStopping`.

---

## 📊 Dataset Overview

The dataset is partitioned into three splits containing over **42,000 images**:

| Split | Images Count | Classes | Distribution |
| :--- | :---: | :---: | :--- |
| **Training** | 30,331 | 2 | `wildfire`, `nowildfire` |
| **Validation** | 6,300 | 2 | `wildfire`, `nowildfire` |
| **Test** | 6,316 | 2 | `wildfire`, `nowildfire` |
| **Total** | **42,947** | **2** | **Full Dataset** |

- **Format:** JPG images
- **Native Dimensions:** 350 × 350 pixels (RGB, 3 channels)
- **Target Dimensions for Training:** 349 × 349 × 3

### Directory Structure

```text
Data_set_for_DL/
├── train/
│   ├── wildfire/
│   └── nowildfire/
├── valid/
│   ├── wildfire/
│   └── nowildfire/
└── test/
    ├── wildfire/
    └── nowildfire/
```

> **Note:** The dataset is hosted externally (e.g., Google Drive) and is not bundled directly inside this git repository due to file size. Mount your Google Drive or point `train_dir`, `val_dir`, and `test_dir` to your local folder.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.x
- **Deep Learning Framework:** TensorFlow 2.x, Keras
- **Image Processing:** OpenCV (`cv2`), PIL / Pillow
- **Augmentation:** Albumentations
- **Data Science & Analytics:** NumPy, Pandas, Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab (GPU accelerated)

---

## 🏗️ Model Architecture

The custom CNN (`first_model`) is designed sequentially to extract hierarchical features while controlling parameter bloat:

```text
Model: "sequential"
_________________________________________________________________
 Layer (type)               Output Shape           Param #
=================================================================
 Conv2D (8 filters, 2×2)    (None, 348, 348, 8)        104
 MaxPooling2D (2×2)          (None, 174, 174, 8)          0

 Conv2D (16 filters, 2×2)   (None, 173, 173, 16)        528
 MaxPooling2D (2×2)          (None,  86,  86, 16)          0

 Conv2D (32 filters, 2×2)   (None,  85,  85, 32)      2,080
  └─ L2 Regularizer (decay=1e-3)
 MaxPooling2D (2×2)          (None,  42,  42, 32)          0

 Flatten                     (None, 56448)                  0
 Dense (300 units, ReLU)     (None, 300)           16,934,700
 Dropout (rate=0.5)          (None, 300)                    0
 Dense (100 units, ReLU)     (None, 100)               30,100
 Dropout (rate=0.4)          (None, 100)                    0
 Dense (2 units, Softmax)    (None, 2)                    202
=================================================================
Total params:      16,967,714 (64.73 MB)
Trainable params:  16,967,714 (64.73 MB)
Non-trainable:     0
```

### Architectural Highlights
1. **Hierarchical Feature Extraction:** Starts with small receptive fields (2×2 filters) progressing from 8 to 32 feature maps.
2. **L2 Weight Regularization:** Integrated on deeper convolutional layers to penalize excessively large weights.
3. **Dual Dropout Regularization:** High dropout rates (50% and 40%) in the dense classification head to prevent co-adaptation of neurons.

---

## ⚙️ Hyperparameters & Training Setup

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Input Shape** | `(349, 349, 3)` | Normalized RGB inputs |
| **Batch Size** | `256` | Large batch training for stability |
| **Max Epochs** | `10` | Early stopping halts training if validation plateaus |
| **Optimizer** | `Adam` | Initial learning rate: `1e-5`, weight decay: `1e-5` |
| **Loss Function** | `Categorical Crossentropy` | Softmax binary probability distribution |
| **Monitored Metrics** | `AUC`, `Accuracy` | Evaluates classification separation |
| **Callbacks** | `ModelCheckpoint`, `EarlyStopping` | Monitors `val_loss`, saves `first_model.hdf5` |

---

## 🔄 End-to-End Pipeline

```
  ┌────────────────────────┐
  │  1. Dataset Discovery  │ ──► Glob paths & extract labels into Pandas DataFrames
  └────────────────────────┘
              │
              ▼
  ┌────────────────────────┐
  │ 2. Visual EDA & Stats  │ ──► Dimensions validation & 6-sample visual verification
  └────────────────────────┘
              │
              ▼
  ┌────────────────────────┐
  │ 3. Data Augmentation   │ ──► Albumentations pipeline (Flips, Rotations up to ±75°)
  └────────────────────────┘
              │
              ▼
  ┌────────────────────────┐
  │ 4. Flow Generators     │ ──► Rescaled (1/255) flow_from_dataframe generators
  └────────────────────────┘
              │
              ▼
  ┌────────────────────────┐
  │ 5. CNN Model Training  │ ──► Compile Adam, train with Checkpoint & EarlyStopping
  └────────────────────────┘
              │
              ▼
  ┌────────────────────────┐
  │ 6. Model Evaluation    │ ──► Test Accuracy, Precision, Recall, F1 & Confusion Matrix
  └────────────────────────┘
              │
              ▼
  ┌────────────────────────┐
  │ 7. Visual Diagnostics  │ ──► Loss & AUC Epoch Curves + Qualitative Image Inferences
  └────────────────────────┘
```

---

## 🧪 Model Evaluation & Results

The notebook includes full post-training evaluation on the **6,316 unseen test images**:

1. **Model Checkpoint Restoration:** Loads the top-performing checkpoint (`first_model.hdf5`).
2. **Quantitative Metrics:**
   - Overall Test Accuracy
   - Precision, Recall, and F1-Score breakdown for both `wildfire` and `nowildfire`
3. **Confusion Matrix:** Heatmap visualising True Positives, False Positives, True Negatives, and False Negatives.
4. **Learning Dynamics:** Side-by-side training vs. validation loss and AUC curve plots.
5. **Visual Verification:** 3×3 grid of random test images color-coded by prediction correctness (Green = Correct, Red = Incorrect).

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/Suhirdha24/Wildfire-Prediction-using-Deep-Learning-Algorithm.git
cd Wildfire-Prediction-using-Deep-Learning-Algorithm
```

### 2. Install Dependencies
```bash
pip install numpy pandas opencv-python matplotlib seaborn tensorflow scikit-learn albumentations pillow jupyter
```

### 3. Open Notebook
Launch the notebook in your local Jupyter environment or upload it directly to [Google Colab](https://colab.research.google.com/):
```bash
jupyter notebook DL_project_2.ipynb
```

### 4. Configure Paths & Run
Ensure your dataset paths point to your data directory:
```python
train_dir = Path('/content/drive/MyDrive/Data_set_for_DL/train')
test_dir  = Path('/content/drive/MyDrive/Data_set_for_DL/test')
val_dir   = Path('/content/drive/MyDrive/Data_set_for_DL/valid')
```
Run all cells consecutively. The trained model checkpoint will be saved to `first_model.hdf5`.

---

## 📁 Repository Structure

```text
Wildfire-Prediction-using-Deep-Learning-Algorithm/
├── DL_project_2.ipynb     # Complete notebook: EDA, Training, Evaluation & Visuals
└── README.md              # Project documentation and architectural overview
```

---

## 🔮 Future Roadmap

- [x] **Add Test Set Evaluation:** Accuracy, Precision, Recall, F1-Score & Confusion Matrix *(Completed)*
- [x] **Plot Learning Curves:** Loss & AUC graphs across training epochs *(Completed)*
- [x] **Qualitative Predictions Visualizer:** Grid inference with color-coded labels *(Completed)*
- [ ] **Transfer Learning Exploration:** Benchmark against pre-trained architectures (**MobileNetV2**, **EfficientNetB0**, **ResNet50**)
- [ ] **Interactive Web Demo:** Build a **Streamlit** or **Gradio** web app for instant image drag-and-drop classification
- [ ] **Edge Deployment:** Quantize and convert model to **TensorFlow Lite (`.tflite`)** for embedded drone cameras

---

## 📜 Disclaimer

This project is developed for **academic and educational purposes**. While deep learning models offer powerful predictive capabilities, predictions must be verified by certified fire management tools before making operational disaster response decisions.

---

## 📄 License

This project is open-source and released under the [MIT License](LICENSE).

---

## 👩‍💻 Author

**Suhirdha K S**  
Deep Learning & Computer Vision Enthusiast  
[![GitHub](https://img.shields.io/badge/GitHub-Suhirdha24-181717?style=flat&logo=github)](https://github.com/Suhirdha24)

---

<div align="center">
  <sub>⭐ If you found this project helpful, feel free to give this repository a star!</sub>
</div>
