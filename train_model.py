import torch
from ultralytics import YOLO

print("CUDA available:", torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device name:", device)

# Load a model
model = YOLO("yolov8x.pt")
model.to(device)

project = 'adas.lane_detection'
name = 'yolov8x_640'

# Train the model
results = model.train(
    data="data_single_class.yaml",
    epochs=55,
    batch=12,
    imgsz=640,
    patience=10,
    # cache="disk",
    project=project,
    name=name,
    # rect=True,
    amp=True,
    plots=True,
    # Augmentation
    hsv_h=0.03,
    hsv_s=0.75,
    hsv_v=0.45,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=10,
    perspective=0.001,
    fliplr=0.0,
    bgr=0.01,
    mosaic=0.05,
    erasing=0.3,
    # Utils
    workers=0
)
