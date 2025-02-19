import cv2
import numpy as np
import matplotlib.cm as cm
def xywhn2xywh(bbox, width,height):
    return int((bbox[0]-(bbox[2]/2))*width),int((bbox[1]-(bbox[3]/2))*height), int(bbox[2]*width), int(bbox[3]*height)

def xywh2topleftwh(bbox):
    x = int(bbox[0] - (bbox[2] / 2))
    y = int(bbox[1] - (bbox[3] / 2))
    return x, y, bbox[2], bbox[3]

def read_annotation(f_path):
    with open(f_path) as f:
        lines = f.read().strip().split("\n")
        lines = [x.split(" ") for x in lines]
        data = ([list(map(float, i)) for i in lines])
        data = np.array(data, dtype=np.float32)
        bboxes = data[:, 1:5]
        classes = data[:, 0]
        conf = data[:, 5]
        print(bboxes)
        print(classes)
        print(conf)
    return classes, bboxes, conf

def gen_color_dict():
    # List of COCO classes
    coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
    n = len(coco_classes)
    # Generate color map for the number of COCO classes
    x = np.arange(n)
    ys = [i + x + (i * x) ** 2 for i in range(n)]
    color_map = cm.rainbow(np.linspace(0, 1, len(ys))) * 255
    print(f"COLORS:{color_map}")
    # Create a dictionary mapping class names to colors
    color_dict = {class_name: color_map[i] for i, class_name in enumerate(coco_classes)}
    return color_dict