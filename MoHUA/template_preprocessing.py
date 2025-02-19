import cv2
import os

path = "./templates"
templates = os.listdir(path)

for template in templates:
    filepath = os.path.join(path, template)
    img = cv2.imread(filepath)

    cv2.imshow("Image", img)
    cv2.waitKey(0)