import cv2
import numpy as np
from ultralytics import YOLO

from helper import *
import os
# from deep_sort.deep_sort.tracker import Tracker
# from deep_sort.deep_sort import nn_matching
# from deep_sort.deep_sort.detection import Detection
# from deep_sort.tools import generate_detections as gdet


def number_key(l):
    num = int(l.split(".")[0])
    return num

coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]

color_dict = gen_color_dict()
# Example of how the color dictionary looks
for class_name, color in color_dict.items():
    print(f"{class_name}: {color}")



def drawBox(image, bbox, color=(0,255,0)):
    width = image.shape[1]
    height = image.shape[0]
    print(f"{width}x{height}")
    cx = int(bbox[0]*width)
    cy = int(bbox[1]*height)
    coords = (int((bbox[0]-bbox[2]/2)*width), int((bbox[1]-bbox[3]/2)*height))
    coords2 = (int((bbox[0]+bbox[2]/2)*width), int((bbox[1]+bbox[3]/2)*height))
    image = cv2.drawMarker(image, (cx,cy), color)
    print(f"Point {cx}, {cy}")
    image = cv2.rectangle(image, coords, coords2, color, 2)
    return image

annotation_path = "./annotations/14_h265_20231216184310.mp4_xxx"
HEIGHT = 720
WIDTH = 1080
np.random.seed(1)
file_list = os.listdir(annotation_path)
file_list = sorted(file_list, key=number_key)
annotation = np.random.choice(file_list)
image_name = f"{int(annotation.split('.')[0])}.jpg"
# image_name = f"00001.jpg"
print(image_name)

img = cv2.imread(f"./video/14_h265_20231216184310.mp4_xxx/{image_name}")
img = cv2.resize(img, (WIDTH, HEIGHT))
with open(os.path.join(annotation_path, annotation)) as f:
    classes, bboxes, confs = read_annotation(os.path.join(annotation_path, annotation))
    for (i, box) in enumerate(bboxes):
        print(int(classes[i]))
        img = drawBox(img, box, color=color_dict[coco_classes[int(classes[i])]])
    cv2.imshow("Image",img)
    cv2.waitKey(0)

cv2.imshow("Image", img)
cv2.waitKey(0)