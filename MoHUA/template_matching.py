import cv2
import os
import numpy as np
import pytesseract

vid_path = "../datasets/Patiala CCTV/14_h265_20231216184310.mp4"

cap = cv2.VideoCapture(vid_path)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1080, 720))
    if not ret:
        break

    datetime = frame[0:40, 140:260]
    datetime = cv2.cvtColor(datetime, cv2.COLOR_BGR2GRAY)
    # datetime  = cv2.Canny(datetime, 100, 200)
    ret, datetime = cv2.threshold(datetime, 250, 255, cv2.THRESH_BINARY)
    cv2.imshow("datetime", datetime)
    text = pytesseract.image_to_string(datetime)
    text = text.strip()
    print(text)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
