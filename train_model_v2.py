import os
import random
import shutil
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class AircraftDetector:
    def __init__(self):
        self.project_dir = os.path.abspath(os.path.dirname(__file__))
        self.dataset_dir = os.path.join(self.project_dir, 'dataset')
        self.model = None

        # split dataset (train, test, val)
        self.split_dataset()
        self.create_dataset_yaml()

    def split_dataset(self):
        img_dir = os.path.join(self.dataset_dir, 'images/train')
        lbl_dir = os.path.join(self.dataset_dir, 'labels/train')

        # new directories
        new_dirs = {
            'train': {'images': 'dataset/images/train', 'labels': 'dataset/labels/train'},
            'val': {'images': 'dataset/images/val', 'labels': 'dataset/labels/val'},
            'test': {'images': 'dataset/images/test', 'labels': 'dataset/labels/test'}
        }

        for split in new_dirs.values():
            for key, path in split.items():
                os.makedirs(path, exist_ok=True)

        # shuffle 
        images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        random.shuffle(images)

        # split 
        total = len(images)
        train_split = int(0.8 * total)
        val_split = int(0.9 * total)

        splits = {
            'train': images[:train_split],
            'val': images[train_split:val_split],
            'test': images[val_split:]
        }

        # Move imgs
        for split, imgs in splits.items():
            for img in imgs:
                img_path = os.path.join(img_dir, img)
                lbl_path = os.path.join(lbl_dir, img.replace('.jpg', '.txt'))

                if os.path.exists(lbl_path):
                    shutil.move(img_path, os.path.join(new_dirs[split]['images'], img))
                    shutil.move(lbl_path, os.path.join(new_dirs[split]['labels'], img.replace('.jpg', '.txt')))

        print("Dataset split into train, val, and test.")

    # yaml file
    def create_dataset_yaml(self):
        yaml_content = f"""
path: {self.dataset_dir}
train: {self.dataset_dir}/images/train
val: {self.dataset_dir}/images/val
test: {self.dataset_dir}/images/test

nc: 1
names: ['aircraft']
"""
        yaml_path = os.path.join(self.project_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())

        print(f"Created dataset.yaml at: {yaml_path}")

    # training yolov8 nano model
    def train_model(self):
        self.model = YOLO('yolov8n.pt')
        yaml_path = os.path.join(self.project_dir, 'dataset.yaml')

        self.model.train(
            data=yaml_path,
            epochs=15,
            imgsz=640,
            batch=16,
            name='aircraft_detector'
        )
        print("Training complete.")

if __name__ == "__main__":
    detector = AircraftDetector()
    detector.train_model()
