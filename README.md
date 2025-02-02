# Aircraft Detector using YOLOv8

## Overview
This project implements an **Aircraft Detector** using **YOLOv8** to detect aircraft in images. The dataset is automatically split into **train, validation, and test sets**, and a YOLOv8 **Nano model** is trained on this data. 

Examples: 
![1](https://github.com/agneya-1402/Aircraft_Detector/blob/main/demo_outputs/Figure_1.png)
![2](https://github.com/agneya-1402/Aircraft_Detector/blob/main/demo_outputs/Figure_11.png)
![3](https://github.com/agneya-1402/Aircraft_Detector/blob/main/demo_outputs/Figure_2.png)
![4](https://github.com/agneya-1402/Aircraft_Detector/blob/main/demo_outputs/Figure_3.png)
![5](https://github.com/agneya-1402/Aircraft_Detector/blob/main/demo_outputs/Figure_5.png)

## Project Structure
```
├── dataset.yaml  # YOLO dataset config
├── weights/
│   ├── best.pt  # Best trained model
│   ├── last.pt  # Last trained model
├── train_model_v2.py  # Training script
├── test_v2.py  # Inference script
├── README.md  # Project documentation
```

## Installation
Ensure you have Python 3.8+ installed. Then, install dependencies:
```bash
pip install ultralytics numpy matplotlib opencv-python 
```

## Dataset Preparation
The dataset is automatically split into **train (80%)**, **validation (10%)**, and **test (10%)** during execution.
Dataset: https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset/data

## Training the Model
To train the YOLOv8 model, run:
```bash
python train_model_v2.py
```
This will:
- Load the **YOLOv8-Nano** model (`yolov8n.pt`)
- Train the model for **15 epochs** (increase to 50 if your machine can handle)
- Save trained weights in the `weights/` directory

## Testing the Model
Run inference on test images:
```bash
python test_v2.py
```
This will:
- Detect objects in all test images
- Draw bounding boxes and count objects
- Display results using Matplotlib

## Evaluation Metrics
The trained model is evaluated using:
- **Precision**
- **Recall**
- **F1 Score**

Results : ![Results](https://github.com/agneya-1402/Aircraft_Detector/blob/main/outputs/results.png)
F1 Curve : ![F1_curve](https://github.com/agneya-1402/Aircraft_Detector/blob/main/outputs/F1_curve.png)
PRECISION Curve : ![PR_curve](https://github.com/agneya-1402/Aircraft_Detector/blob/main/outputs/P_curve.png)
PRECISION RECALL Curve : ![P_curve](https://github.com/agneya-1402/Aircraft_Detector/blob/main/outputs/PR_curve.png)
RECALL Curve : ![R_curve](https://github.com/agneya-1402/Aircraft_Detector/blob/main/outputs/R_curve.png)


## Model Weights
The trained YOLOv8 model weights are saved in the `weights/` directory:
- **best.pt** – Best performing model checkpoint
- **last.pt** – Model from the last training epoch

## Author
**Agneya Pathare** – Robotics Engineer | AI & Computer Vision Developer


