import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # or use 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'

# Train the model
model.train(
    data='disease.yaml',  # Path to your dataset YAML file
    epochs=50,                         # Number of epochs
    batch=16,                          # Batch size
    imgsz=640,                         # Image size (YOLOv8 standard is 640)
    project='plant_disease_project',    # Project name
    name='yolov8_finetuned',            # Run name
    exist_ok=True                       # Overwrite existing project
)
