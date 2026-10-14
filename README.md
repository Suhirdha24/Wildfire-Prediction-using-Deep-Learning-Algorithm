# Wildfire Detection Using Deep Learning

## Overview

Wildfires pose a serious threat to forests, wildlife, the environment, and human lives. Early detection can help support faster response and reduce potential damage.

This project implements an image classification model using **Convolutional Neural Networks (CNNs)** to distinguish between wildfire and non-wildfire images. Developed using Python and TensorFlow/Keras, the project includes image dataset preparation, visualization, data augmentation, model training, and validation.

## Objectives

- Classify images into `wildfire` and `nowildfire` categories.
- Prepare and organize image datasets for deep learning.
- Apply image augmentation to improve training data diversity.
- Build a CNN model for binary image classification.
- Train the model using training and validation datasets.
- Save the best-performing model checkpoint during training.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Image Processing:** OpenCV, Pillow
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Data Augmentation:** Albumentations, Keras ImageDataGenerator
- **Development Environment:** Jupyter Notebook / Google Colab

## Project Workflow

1. **Dataset Preparation:** Load image paths from the training, validation, and testing directories and generate their corresponding labels.
2. **Exploratory Data Analysis:** Inspect the dataset, identify image dimensions and class labels, and visualize sample images.
3. **Image Augmentation:** Apply flipping and rotation to introduce variations into the training images.
4. **Preprocessing:** Resize images to the configured target dimensions and normalize pixel values to the range 0–1.
5. **CNN Model Development:** Build a sequential CNN architecture with convolutional layers, max pooling, fully connected layers, and dropout.
6. **Model Training:** Train the network using the Adam optimizer and categorical cross-entropy loss.
7. **Model Checkpointing:** Save the best model checkpoint based on validation loss and use early stopping to help limit unnecessary training.

## Model Architecture

The CNN model includes:

- Convolutional layers with ReLU activation
- Max-pooling layers for spatial feature reduction
- Flatten layer to transform feature maps into a vector
- Fully connected dense layers
- Dropout layers to help reduce overfitting
- Softmax output layer with two classes

## Dataset Structure

Organize your dataset in the following format:

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

The notebook expects JPG images inside these directories. The dataset is not included in this repository unless you upload it separately.

## Installation

Install the required libraries:

```bash
pip install numpy pandas opencv-python matplotlib seaborn tensorflow scikit-learn albumentations pillow jupyter
```

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/Wildfire-Detection-Using-Deep-Learning.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Wildfire-Detection-Using-Deep-Learning
   ```

3. Install the dependencies listed above.

4. Download or provide access to a wildfire image dataset organized according to the directory structure described above.

5. Open `DL_project_2.ipynb` in Jupyter Notebook or Google Colab.

6. Update the dataset paths in the notebook to match your environment. If using Google Colab, mount Google Drive and update the paths accordingly.

7. Run the notebook cells in order to prepare the dataset, visualize images, perform augmentation, build the CNN, and train the model.

## Model Training Configuration

The current notebook uses the following configuration:

| Parameter | Value |
|---|---|
| Input image target size | 350 × 350 |
| Input channels | 3 (RGB images) |
| Batch size | 256 |
| Maximum epochs | 10 |
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |
| Output classes | 2 |
| Early stopping patience | 10 epochs |
| Model checkpoint | `first_model.hdf5` |

## Expected Output

After running the notebook, the training process produces:

- Visualizations of wildfire and non-wildfire images
- Examples of original and augmented images
- CNN model architecture summary
- Training and validation progress
- A saved model checkpoint named `first_model.hdf5`, when checkpoint saving succeeds

Actual classification performance should be measured using a separate evaluation on the test dataset. Accuracy, precision, recall, F1-score, and a confusion matrix can be reported once those metrics have been calculated.

## Applications

- Forest fire monitoring research
- Image-based wildfire screening
- Environmental monitoring systems
- Computer vision and deep learning research
- Prototyping early wildfire warning solutions

## Limitations

- Performance depends on dataset quality, diversity, and class balance.
- The model requires further testing on unseen images and real-world conditions.
- Image classification alone does not establish the geographic location or severity of a fire.
- This project is a prototype and should not be treated as a standalone operational wildfire alert system without additional validation.

## Future Enhancements

- Experiment with transfer learning architectures such as MobileNetV2, EfficientNet, or ResNet.
- Evaluate performance using precision, recall, F1-score, and a confusion matrix.
- Build an interface using Streamlit for image uploads and predictions.
- Explore real-time video or camera-based wildfire detection.
- Optimize the trained model for deployment on edge devices.

## Disclaimer

This project is intended for educational and research purposes. Predictions should be validated before being used in any real-world fire monitoring or emergency response application.

## License

You may add an appropriate open-source license, such as the MIT License, if you intend to allow others to reuse and modify this project.
