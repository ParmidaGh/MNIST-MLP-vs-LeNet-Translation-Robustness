<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:172554,50:2563eb,100:06b6d4&height=220&section=header&text=MNIST%20Classification%20with%20MLP%20and%20LeNet&fontSize=32&fontColor=ffffff&fontAlignY=50&animation=fadeIn"/>
</div>

# MNIST Classification with MLP and LeNet: Translation Robustness Analysis

This project presents a comparative deep learning study of a **Multilayer Perceptron (MLP)** and **LeNet-style Convolutional Neural Network (CNN)** for handwritten digit classification on the MNIST dataset. Beyond standard classification accuracy, the project investigates how both architectures respond to controlled **spatial translations of input images**, providing an empirical perspective on the effect of architectural inductive biases on translation robustness.

<div align="left">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat\&logo=python\&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?style=flat\&logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Neural_Networks-D00000?style=flat\&logo=keras\&logoColor=white)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=flat\&logo=numpy\&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Image_Transformation-8CAAE6?style=flat\&logo=scipy\&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat\&logo=matplotlib\&logoColor=white)](https://matplotlib.org/)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-111827?style=flat)](http://yann.lecun.com/exdb/mnist/)
[![Deep Learning](https://img.shields.io/badge/Domain-Deep_Learning-7C3AED?style=flat)](#)
[![Computer Vision](https://img.shields.io/badge/Domain-Computer_Vision-0891B2?style=flat)](#)
[![License](https://img.shields.io/badge/License-MIT-4B5563?style=flat)](https://opensource.org/licenses/MIT)

</div>

## Abstract

Robustness to spatial transformations is an important property of image classification models. While convolutional architectures are explicitly designed to exploit local spatial structure, fully connected networks operate on flattened representations and therefore have a fundamentally different inductive bias.

This project compares two neural network architectures for handwritten digit recognition on the **MNIST** dataset: a fully connected **Multilayer Perceptron (MLP)** and a classical **LeNet-style CNN**.

Both models are trained for 15 epochs using the Adam optimizer and categorical cross-entropy loss. Their standard classification performance is then evaluated on the MNIST test set. To investigate translation robustness, selected test images are systematically shifted in different spatial directions and magnitudes using controlled image transformations. The predictions produced by both models are subsequently compared across the transformed inputs.

The experiments provide a practical demonstration of the relationship between **model architecture, spatial inductive bias, and robustness to input translations**.

## Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Model Architectures](#-model-architectures)
4. [Translation Robustness Analysis](#-translation-robustness-analysis)
5. [Experimental Results](#-experimental-results)
6. [Technologies Used](#-technologies-used)
7. [Repository Structure](#-repository-structure)
8. [Installation](#-installation)
9. [Usage](#-usage)
10. [Author](#-author)
11. [Support](#-support)
12. [License](#license)

# Overview

The project follows a two-stage experimental pipeline:

1. Train an **MLP classifier** directly on normalized 28×28 MNIST images.
2. Train a **LeNet-style CNN** using the same dataset with an explicit channel dimension.
3. Evaluate both models on the original MNIST test set.
4. Select representative test images.
5. Apply controlled spatial translations to the selected images.
6. Visualize the original and transformed samples.
7. Generate predictions from both trained models.
8. Compare model behavior under different translation conditions.

The experiment is designed to highlight the difference between a model that processes pixels primarily as independent input features and a convolutional model that exploits local spatial patterns.

# Key Features

* MNIST handwritten digit classification
* Comparative evaluation of MLP and LeNet architectures
* End-to-end training with TensorFlow/Keras
* GPU-based model training
* Input normalization and categorical encoding
* Controlled spatial image translation
* Visualization of original and shifted images
* Prediction analysis under multiple translation configurations
* Empirical investigation of translation robustness
* Reproducible notebook-based experimentation

# Model Architectures

## Multilayer Perceptron

The MLP processes each 28×28 MNIST image as a flattened feature representation.

The architecture consists of:

| Layer   | Configuration           |
| :------- | :----------------------- |
| Input   | 28 × 28 grayscale image |
| Flatten | 784 input features      |
| Dense   | 512 units, ReLU         |
| Dense   | 256 units, ReLU         |
| Output  | 10 units, Softmax       |

The model is trained using:

* **Optimizer:** Adam
* **Loss:** Categorical Cross-Entropy
* **Epochs:** 15
* **Batch Size:** 64
* **Validation Split:** 10%

## LeNet

The CNN follows a classical LeNet-style architecture that preserves the spatial structure of the input throughout the convolutional feature extraction stages.

| Layer       | Configuration               |
| :----------- | :--------------------------- |
| Input       | 28 × 28 × 1 grayscale image |
| Convolution | 6 filters, 5 × 5, Sigmoid   |
| Pooling     | 2 × 2 Average Pooling       |
| Convolution | 16 filters, 5 × 5, Sigmoid  |
| Pooling     | 2 × 2 Average Pooling       |
| Flatten     | Feature vector              |
| Dense       | 120 units, Sigmoid          |
| Dense       | 84 units, Sigmoid           |
| Output      | 10 units, Softmax           |

The LeNet model uses the same optimization and training configuration as the MLP to make the architectural comparison more consistent.

# Translation Robustness Analysis

## Motivation

A handwritten digit may remain visually recognizable after a small spatial displacement, but a model's prediction can still change depending on how it represents spatial information.

To investigate this behavior, the project applies controlled translations to selected MNIST test images using `scipy.ndimage.shift`.

Each transformation is defined by a two-dimensional shift:

```text
(row_shift, column_shift)
```

Positive and negative values are used to move the digit in different spatial directions.

## Experimental Procedure

For each selected test image:

1. The original digit is extracted from the MNIST test set.
2. Multiple spatial translations are generated.
3. The transformed images are displayed alongside the original image.
4. The MLP receives the shifted images in 28×28 format.
5. The LeNet model receives the same images in 28×28×1 format.
6. Predictions from both models are recorded.
7. The prediction behavior is compared across different translations.

This setup provides a simple controlled experiment for examining how changes in spatial position affect classification behavior.

# Experimental Results

Both architectures achieve high classification accuracy on the standard MNIST test set.

| Model | Test Accuracy |
| :----- | :------------: |
| MLP   |    **97.79%** |
| LeNet |    **98.52%** |

The results show that the LeNet architecture achieves higher standard test accuracy in this experiment.

However, the primary purpose of the project is not limited to comparing test accuracy. The additional translation experiments examine whether predictions remain stable when the same handwritten digit is spatially displaced.

The notebook evaluates multiple translated samples using different shift configurations, allowing the behavior of the two architectures to be inspected directly rather than relying only on aggregate accuracy.

## Observed Prediction Behavior

The translation experiments demonstrate that spatial transformations can influence model predictions even when the underlying digit remains recognizable.

The analysis therefore illustrates an important distinction:

* **MLP:** Processes flattened pixel values and does not explicitly encode spatial locality.
* **LeNet:** Uses convolution and pooling to extract local spatial features and introduces stronger architectural assumptions about image structure.

These differences make the two models useful for studying how neural network architecture affects behavior under input transformations.

# Technologies Used

| Category                     | Tools                |
| :---------------------------- | :-------------------- |
| Programming Language         | Python               |
| Deep Learning Framework      | TensorFlow / Keras   |
| Neural Network Architectures | MLP, LeNet-style CNN |
| Dataset                      | MNIST                |
| Numerical Computing          | NumPy                |
| Image Transformation         | SciPy                |
| Visualization                | Matplotlib           |
| Hardware Acceleration        | GPU via TensorFlow   |

# Repository Structure

```text
MNIST-MLP-vs-LeNet-Translation-Robustness/
│
├── MNIST-MLP-vs-LeNet-Translation-Robustness-Analysis.ipynb
│
├── LICENSE
│
└── README.md
```

# Installation

## Clone Repository

```bash
git clone https://github.com/ParmidaGh/MNIST-MLP-vs-LeNet-Translation-Robustness.git

cd MNIST-MLP-vs-LeNet-Translation-Robustness
```

## Create Environment

```bash
conda create -n mnist-robustness python=3.10

conda activate mnist-robustness
```

## Install Dependencies

```bash
pip install tensorflow numpy scipy matplotlib jupyter
```

# Usage

Launch the Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
MNIST-MLP-vs-LeNet-Translation-Robustness-Analysis.ipynb
```

The notebook automatically downloads the MNIST dataset through TensorFlow/Keras, preprocesses the images, trains both models, evaluates their test accuracy, generates spatially shifted samples, visualizes the transformations, and produces predictions for the transformed images.

## GPU Training

The training cells are configured to execute the model training and evaluation using:

```python
with tf.device('/GPU:0'):
```

Therefore, a compatible TensorFlow GPU environment can be used to accelerate the experiments.

# Author

**Parmida Ghamari**

Research Interests: **Deep Learning, Computer Vision, Neural Networks, Representation Learning, Machine Learning, Artificial Intelligence, Robustness of Deep Learning Models, and Applied AI**

📧 [Parmida.ghamari@gmail.com](mailto:Parmida.ghamari@gmail.com)
💻 [github.com/ParmidaGh](https://github.com/ParmidaGh)
💼 [linkedin.com/in/parmida-ghamari](https://www.linkedin.com/in/parmida-ghamari)

# ⭐ Support

If you find this project useful, consider giving it a star.

# License

This project is licensed under the MIT License.

---

<p align="center">
Built using TensorFlow, Keras, NumPy, SciPy, and Matplotlib
</p>
