import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


MODEL_PATH = "runs/detect/aircraft_detector3/weights/best.pt"  
TEST_IMAGES_DIR = "demo_inputs" 

# Load Model
model = YOLO(MODEL_PATH)

# loop thru demo imgs
for img_name in os.listdir(TEST_IMAGES_DIR):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping invalid image: {img_path}")
        continue

    # run model on img
    results = model(img_path)

    # Results
    for result in results:
        num_targets = len(result.boxes)  # Number of detected targets
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordinates
            conf = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            # Draw Box 
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Class & Confidence
            label = f"ID {class_id}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected {num_targets} objects in {img_name}")
    plt.axis("off")
    plt.show()
