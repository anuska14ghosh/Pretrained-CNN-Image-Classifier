# Transfer Learning for Computer Vision using PyTorch

This project demonstrates how to use **Transfer Learning** in PyTorch for image classification tasks using pretrained convolutional neural networks (CNNs). The implementation uses the popular **ResNet18** architecture pretrained on the ImageNet dataset to classify images of ants and bees with high accuracy, even on a small dataset.

The project covers two major transfer learning approaches:

* **Fine-Tuning a Pretrained Model**
  The entire pretrained ResNet18 model is loaded, and the final fully connected layer is replaced to adapt the network for binary classification. The model is then trained on the custom dataset while updating all network weights.

* **Feature Extraction using Frozen Layers**
  All pretrained layers are frozen, and only the final classification layer is trained. This reduces training time and computational cost while still achieving strong performance.

## Features

* Transfer learning using pretrained ResNet18
* Image classification with PyTorch
* Data augmentation and normalization
* GPU/CPU accelerator support
* Model training and validation pipeline
* Learning rate scheduling
* Best model checkpoint saving
* Visualization of predictions
* Custom image inference support
* Fine-tuning and feature extraction comparison

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* PIL (Python Imaging Library)

## Dataset

The project uses the **Hymenoptera dataset** containing images of:

* Ants
* Bees

The dataset is a small subset of ImageNet and is ideal for demonstrating transfer learning techniques on limited data.

## Project Workflow

1. Load and preprocess image data
2. Apply data augmentation techniques
3. Initialize pretrained ResNet18
4. Replace the final classification layer
5. Train the model using transfer learning
6. Evaluate performance on validation data
7. Visualize predictions on sample images
8. Perform inference on custom images

## Key Concepts Demonstrated

* Transfer Learning
* Fine-Tuning
* Feature Extraction
* CNN-based Image Classification
* Learning Rate Scheduling
* Model Evaluation
* Data Augmentation

## Results

The pretrained model achieves strong classification performance despite the small dataset size, demonstrating the effectiveness of transfer learning for computer vision tasks with limited training data.

## Future Improvements

* Add support for more architectures (VGG, EfficientNet, Vision Transformers)
* Deploy using Streamlit or Flask
* Add real-time webcam inference
* Experiment with larger datasets
* Hyperparameter optimization
* Multi-class classification support

## How to Run

### Install Dependencies

```bash
pip install torch torchvision matplotlib numpy pillow
```

### Run the Project

```bash
python transfer_learning_tutorial.py
```

## Learning Outcome

This project provides a practical understanding of how pretrained deep learning models can be reused for custom computer vision applications, significantly reducing training time and improving performance on small datasets.
