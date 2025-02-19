from ultralytics import YOLO

import numpy as np
import os
import cv2

model = YOLO("./weights/yolov8l.pt")
filename = "14_h265_20231216184310.mp4_xxx"
vid = cv2.VideoCapture("../datasets/Patiala CCTV/14_h265_20231216184310.mp4")
annotation_path = f"./annotations/{filename}/"
video_path = f"./video/{filename}/"
os.makedirs(video_path)
os.makedirs(annotation_path)
# model.predict(vid, save=True,save_txt=True, save_conf=True,half=True, agnostic_nms=True,classes=[0,1,2,3,5,7])
BATCH_SIZE = 32
frames = []
i = 0
while vid.isOpened():
    ret, frame=  vid.read()
    if not ret:
        break
    if(len(frames) != BATCH_SIZE):
        frames.append(frame)
        continue
    results = model.predict(frames, classes=[0,1,2,3,5,7])
    frames = []
    for result in results:
        img = result.orig_img
        height, width = img.shape[:2]
        img = cv2.resize(img, (width//2, height//2))
        cv2.imwrite(f"{video_path}{i}.jpg", img)
        result.save_txt(f"{annotation_path}{i}.txt", save_conf=True)
        i+=1

    if(i%100 == 0):
        print(f"Processed {i} frames")
    
