# Aircraft Detector using YOLOv8

## Overview
This project implements an **Aircraft Detector** using **YOLOv8** to detect aircraft in images. The dataset is automatically split into **train, validation, and test sets**, and a YOLOv8 **Nano model** is trained on this data. The repository contains the training, testing, dataset files, trained model weights, and evaluation metrics.

Examples:

## Project Structure
```
├── dataset/
│   ├── images/
│   │   ├── train/  # Training images
│   │   ├── val/    # Validation images
│   │   ├── test/   # Test images
│   ├── labels/
│   │   ├── train/  # Training labels
│   │   ├── val/    # Validation labels
│   │   ├── test/   # Test labels
│   ├── dataset.yaml  # YOLO dataset configuration
├── weights/
│   ├── best.pt  # Best trained model
│   ├── last.pt  # Last trained model
├── train.py  # Training script
├── demo_test.py  # Inference script
├── README.md  # Project documentation
```

## Installation
Ensure you have Python 3.8+ installed. Then, install dependencies:
```bash
pip install ultralytics numpy matplotlib opencv-python 
```

## Dataset Preparation
The dataset is automatically split into **train (80%)**, **validation (10%)**, and **test (10%)** during execution.

## Training the Model
To train the YOLOv8 model, run:
```bash
python train.py
```
This will:
- Load the **YOLOv8-Nano** model (`yolov8n.pt`)
- Train the model for **15 epochs**
- Save trained weights in the `weights/` directory

## Testing the Model
Run inference on test images:
```bash
python demo_test.py
```
This will:
- Detect objects in all test images
- Draw bounding boxes and count objects
- Display results using Matplotlib

## Evaluation Metrics
The trained model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Results :
F1 Curve :
PRECISION Curve :
PRECISION RECALL Curve : 
RECALL Curve :


## Model Weights
The trained YOLOv8 model weights are saved in the `weights/` directory:
- **best.pt** – Best performing model checkpoint
- **last.pt** – Model from the last training epoch

## Author
**Agneya Pathare** – Robotics Engineer | AI & Computer Vision Developer


