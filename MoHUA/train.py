from ultralytics import YOLO
import numpy as np

yaml_path = "../datasets/dhaka_traffic_dataset/traffic_update.yaml"

EPOCHS = 10

model = YOLO("./weights/yolov8n.pt")
model.train(data=  yaml_path, epochs=EPOCHS, batch=16)
