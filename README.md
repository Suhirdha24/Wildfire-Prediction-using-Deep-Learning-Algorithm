# Wildfire Prediction Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-CNN-D00000?style=flat-square&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A deep learning-based image classification system to detect wildfires using Convolutional Neural Networks (CNNs).

---

## Overview

Wildfires pose a serious threat to forests, wildlife, the environment, and human lives. Early and accurate detection is critical to support faster emergency response and minimize damage.

This project implements a **binary image classification model** using a custom **Convolutional Neural Network (CNN)** built with TensorFlow/Keras to distinguish between **wildfire** and **non-wildfire** images. The full pipeline covers dataset preparation, exploratory data analysis (EDA), real-time data augmentation, model architecture design, training with callbacks, and checkpoint saving.

---

## Objectives

- Classify images into `wildfire` and `nowildfire` categories
- Prepare and organize a large-scale image dataset for deep learning
- Apply real-time image augmentation (flipping & rotation) to improve model generalization
- Design and train a custom CNN architecture from scratch
- Monitor training performance using AUC and accuracy metrics
- Automatically save the best-performing model checkpoint during training

---

## Dataset Statistics

| Split      | Total Images | Classes |
|------------|:------------:|:-------:|
| Training   | 30,331       | 2       |
| Validation | 6,300        | 2       |
| Test       | 6,316        | 2       |
| **Total**  | **42,947**   | **2**   |

- **Classes:** `wildfire`, `nowildfire`
- **Image Dimensions:** 350 x 350 pixels, 3 channels (RGB)
- **Format:** JPG

### Dataset Directory Structure

```
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

> **Note:** The dataset is hosted on Google Drive and is not included in this repository. Mount your Google Drive in Colab and update the dataset paths accordingly.

---

## Technologies Used

| Category             | Tools / Libraries                                              |
|----------------------|----------------------------------------------------------------|
| Programming Language | Python 3.x                                                     |
| Deep Learning        | TensorFlow 2.x, Keras                                          |
| Image Processing     | OpenCV (cv2), Pillow                                           |
| Data Handling        | NumPy, Pandas, Pathlib                                         |
| Visualization        | Matplotlib, Seaborn                                            |
| Data Augmentation    | Albumentations, Keras ImageDataGenerator                       |
| ML Utilities         | Scikit-learn (classification report, confusion matrix, AUC)    |
| Development          | Google Colab / Jupyter Notebook                                |

---

## Model Architecture

The custom CNN (`first_model`) is a Sequential model with **~16.97 million trainable parameters (64.73 MB)**:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)               Output Shape           Param #
=================================================================
 Conv2D (8 filters, 2x2)    (None, 348, 348, 8)        104
 MaxPooling2D (2x2)          (None, 174, 174, 8)          0

 Conv2D (16 filters, 2x2)   (None, 173, 173, 16)        528
 MaxPooling2D (2x2)          (None,  86,  86, 16)          0

 Conv2D (32 filters, 2x2)   (None,  85,  85, 32)      2,080
  [+ L2 Regularization]
 MaxPooling2D (2x2)          (None,  42,  42, 32)          0

 Flatten                     (None, 56448)                  0
 Dense (300 units, ReLU)     (None, 300)           16,934,700
 Dropout (rate=0.5)          (None, 300)                    0
 Dense (100 units, ReLU)     (None, 100)               30,100
 Dropout (rate=0.4)          (None, 100)                    0
 Dense (2 units, Softmax)    (None, 2)                    202
=================================================================
Total params:      16,967,714  (64.73 MB)
Trainable params:  16,967,714  (64.73 MB)
Non-trainable params:        0
```

**Key design decisions:**
- **L2 Regularization** (`weight_decay = 1e-3`) on the third Conv2D layer to penalize large weights
- **Dropout layers** (50% after Dense-300, 40% after Dense-100) for regularization
- **Softmax** output for probabilistic binary classification

---

## Training Configuration

| Parameter              | Value                                                  |
|------------------------|--------------------------------------------------------|
| Input image size       | 349 x 349 x 3                                          |
| Batch size             | 256                                                    |
| Maximum epochs         | 10                                                     |
| Optimizer              | Adam (learning_rate=1e-5, weight_decay=1e-5)           |
| Loss function          | Categorical Cross-Entropy                              |
| Metrics                | AUC, Accuracy                                          |
| Output classes         | 2 (wildfire, nowildfire)                               |
| Early stopping monitor | val_loss (patience = 10 epochs)                        |
| Model checkpoint       | first_model.hdf5 (save best only)                      |

---

## Project Workflow

```
1. Dataset Loading  -->  2. EDA & Stats  -->  3. Image Visualization
                                                      |
                                                      v
6. CNN Training  <--  5. Data Generators  <--  4. Augmentation
       |
       v
7. Checkpoint Saving (first_model.hdf5)
```

### Step-by-Step Breakdown

1. **Dataset Preparation** — Recursively glob `.jpg` images from `train/`, `valid/`, and `test/` directories; extract folder names as class labels and build Pandas DataFrames.
2. **EDA** — Print dataset statistics (image count, number of classes) and verify image dimensions (350x350x3).
3. **Image Visualization** — Display 6-image grids for each class (`wildfire` and `nowildfire`) using Matplotlib.
4. **Data Augmentation** — Apply an Albumentations pipeline: random horizontal/vertical flip + rotation (limit 75 degrees, always applied). Show side-by-side original vs. augmented comparisons.
5. **Data Generators** — Use `ImageDataGenerator` with per-pixel rescaling (1/255.0) and `flow_from_dataframe` for all three splits. Augmentation is applied to the training generator only.
6. **CNN Model** — Build the Sequential CNN as described in the architecture section above.
7. **Training** — Compile with Adam and categorical cross-entropy; fit for up to 10 epochs with `ModelCheckpoint` and `EarlyStopping` callbacks.

---

## Getting Started

### Prerequisites

- Python 3.x
- Google Colab (recommended) or Jupyter Notebook with GPU support
- Wildfire image dataset organized per the directory structure above

### Installation

```bash
pip install numpy pandas opencv-python matplotlib seaborn tensorflow scikit-learn albumentations pillow jupyter
```

### How to Run

1. **Clone this repository:**

   ```bash
   git clone https://github.com/Suhirdha24/Wildfire-Prediction-using-Deep-Learning-Algorithm.git
   cd Wildfire-Prediction-using-Deep-Learning-Algorithm
   ```

2. **Prepare the dataset** — Organize your wildfire images following the directory structure above. If using Google Colab, upload to Google Drive.

3. **Open the notebook** — Launch `DL_project_2.ipynb` in Google Colab or Jupyter Notebook.

4. **Update dataset paths** to match your environment:

   ```python
   train_dir = Path('/content/drive/MyDrive/Data_set_for_DL/train')
   test_dir  = Path('/content/drive/MyDrive/Data_set_for_DL/test')
   val_dir   = Path('/content/drive/MyDrive/Data_set_for_DL/valid')
   ```

5. **Run all cells in order** — the notebook will prepare data, visualize images, augment, build the CNN, and start training.

6. **Model checkpoint** — The best model weights are automatically saved as `first_model.hdf5`.

---

## Repository Structure

```
Wildfire-Prediction-using-Deep-Learning-Algorithm/
├── DL_project_2.ipynb     # Main Jupyter Notebook (end-to-end pipeline)
└── README.md              # Project documentation
```

> `first_model.hdf5` is generated after training and is not tracked in this repository.

---

## Expected Outputs

After running the full notebook you should see:

- DataFrame previews of train/validation/test splits
- Dataset statistics per split (image count, class distribution)
- 6-image visualization grids for `wildfire` and `nowildfire` classes
- Side-by-side original vs. augmented image pairs
- CNN model architecture summary with layer shapes and parameter counts
- Epoch-by-epoch training logs (loss, AUC, accuracy)
- Saved best model checkpoint: `first_model.hdf5`

---

## Applications

- Forest fire monitoring and early warning research
- Satellite or UAV/drone image-based wildfire screening
- Environmental and climate change monitoring systems
- Computer vision and deep learning education and research
- Prototype backend for wildfire detection mobile or web applications

---

## Limitations

- Model performance is highly dependent on dataset quality, size, and class balance.
- Requires further evaluation on diverse, real-world unseen images.
- Image classification alone cannot determine geographic location or fire severity.
- The model is a research prototype and is not a production-ready fire alert system.
- Training at 349x349 resolution with batch size 256 is GPU-memory intensive.

---

## Future Enhancements

- [ ] Add a full test-set evaluation cell (accuracy, precision, recall, F1-score, confusion matrix)
- [ ] Experiment with transfer learning: MobileNetV2, EfficientNetB0, ResNet50
- [ ] Plot training/validation loss and AUC curves over epochs
- [ ] Build a Streamlit or Gradio web app for drag-and-drop image prediction
- [ ] Explore real-time wildfire detection from video streams or webcam feeds
- [ ] Export and optimize for edge deployment (TensorFlow Lite / ONNX)
- [ ] Experiment with multi-class severity classification (low / medium / high fire risk)

---

## Disclaimer

This project is developed for **educational and research purposes only**. Model predictions should be thoroughly validated before being used in any real-world fire monitoring, public safety, or emergency response application.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Author

**Suhirdha K S**
Deep Learning & Computer Vision Enthusiast
[GitHub Profile](https://github.com/Suhirdha24)

---

If you found this project helpful, please consider giving it a star!
