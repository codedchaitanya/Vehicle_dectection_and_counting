from ultralytics import YOLO

model = YOLO("./runs/detect/train4/weights/last.pt")

metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
