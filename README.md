# SRCNN (Super-Resolution Convolutional Neural Network) using PyTorch

This repository contains an implementation of SRCNN for image super-resolution using PyTorch. SRCNN is a deep learning model designed to enhance the resolution of images while preserving details and reducing computational complexity.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Explanation](#explanation)
- [Results](#results)
  - [Example](#example)   

## Overview
This project implements the SRCNN model in PyTorch, consisting of three convolutional layers. The model is trained on the BSDS500 dataset, employing data augmentation techniques to enhance model generalization. Post-training, the model's performance is evaluated on a validation set using metrics such as PSNR and SSIM.

## Dataset
The BSDS500 dataset is utilized for training and validation. It comprises images with diverse resolutions and content, suitable for super-resolution tasks. The dataset is partitioned into training, validation, and test subsets for respective stages of model training, validation, and evaluation.
For more details, you can access the BSDS500 dataset [here](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500).
## Installation
To run the SRCNN project, follow these steps:

1. Clone the repository:
   ```bash
   https://github.com/AmmarMohamed0/Super-Resolution-CNN.git
   cd srcnn-pytorch
   ```
## Explanation


### Imports and Setup:

- Import necessary libraries including PyTorch, TorchVision, PIL (Python Imaging Library), NumPy, Matplotlib, and skimage.metrics.
- Define the `BSDS500Dataset` class inheriting from `torch.utils.data.Dataset` for handling the BSDS500 dataset.

### Dataset Handling (`BSDS500Dataset` class):

- **`__init__` method**:
  - Initializes with root directory `root_dir` and split (`train`, `val`, `test`).
  - Constructs a list of image files (`image_files`) based on the split.
  - Applies transformations (`transforms`) to the images if provided during initialization.
  
- **`__len__` method**:
  - Returns the total number of images in the dataset.
  
- **`__getitem__` method**:
  - Loads an image from disk based on the index (`idx`), converts it to RGB format using PIL, applies specified transformations, and returns the transformed image.

### Transformations:

- `data_augmentation_transforms`:
  - Composes a series of data augmentation techniques for the training set, including random horizontal flips, rotations, random resized crops, color jittering, and conversion to tensor format.
  
- `standard_transforms`:
  - Defines standard transformations (resize and conversion to tensor) for the validation and test sets.

### Dataset Loading:

- Creates instances of `BSDS500Dataset` for training (`train_dataset`), validation (`val_dataset`), and test (`test_dataset`) sets, specifying respective transformations.

### Data Loading (`DataLoader`):

- Initializes `DataLoader` objects (`train_loader`, `val_loader`, `test_loader`) for iterating through batches of data during training, validation, and testing phases.

### Visualization Functions:

- **`visualize_augmentations` function**:
  - Displays example augmented images from the training dataset.

- **`visualize_progress` function**:
  - Visualizes the input image, low-resolution image (obtained by bicubic interpolation), and output image (enhanced by SRCNN) during model evaluation.

- **`visualize_one_random_image` function**:
  - Selects a random image from the test dataset, applies SRCNN for super-resolution, calculates metrics (PSNR, SSIM, MSE), and displays the original, low-resolution, and enhanced images.

### Model Definition (`SRCNN` class):

- Defines the SRCNN model with three convolutional layers (`conv1`, `conv2`, `conv3`) using `nn.Conv2d` for image super-resolution.
- Uses ReLU activation (`F.relu`) after each convolutional layer.

### Training and Evaluation:

- **Training Loop (`num_epochs` iterations)**:
  - Sets the model to training mode (`model.train()`).
  - Iterates through batches of data (`train_loader`), computes the loss (`MSELoss`), performs backpropagation (`optimizer.step()`), and updates model parameters based on gradients (`optimizer.zero_grad()`).
  - Prints training loss every 10 steps.

- **Validation**:
  - Sets the model to evaluation mode (`model.eval()`).
  

### Results
The trained SRCNN model achieves competitive results in terms of PSNR, SSIM, and MSE compared to state-of-the-art super-resolution models. Visualizations of super-resolution outputs demonstrate the effectiveness of the SRCNN architecture in enhancing image details and quality.
Below are example input-output pairs of the SRCNN model on test images:

### Example 

| ![image alt](https://github.com/AmmarMohamed0/Super-Resolution-CNN/blob/d4a66b4614db1e1e91af4d4f9712d9aa4995f3ba/output.png)


