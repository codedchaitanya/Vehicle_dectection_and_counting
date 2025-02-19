from ultralytics import YOLO
import numpy as np
import cv2
import sys
def drawBox(img, bbox_xywh, color=(0,255,0)):
    # x1, y1, x2, y2 = bbox
    # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    x, y, w, h = bbox_xywh
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    return img

def calcIOU(bb1, bb2):
    """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
    bb1 = bb1.cpu().numpy()[0]
    bb2 = bb2.cpu().numpy()[0]
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def supressDriver(result, thresh=0.3):
    # Remove The person class when IoU high with any vehicle
    vehicle_classes = [1,2,3, 5, 7]
    vehicle_boxes = []
    driver_boxes = []

    for r in result:
        pred_cls = int(r.cls)
        if pred_cls in vehicle_classes:
            vehicle_boxes.append(r)
        elif pred_cls == 0:
            driver_boxes.append(r)
    for vbox in vehicle_boxes:
        for dbox in driver_boxes:
            iou = calcIOU(vbox.xyxy, dbox.xyxy)
            print(f"IoU: {iou}")
            if iou > thresh:
                driver_boxes.remove(dbox)
                print("REMOVING DRIVER BOX")
    vehicle_boxes.extend(driver_boxes)
    return vehicle_boxes

DSET_FOLDER = "../datasets/Patiala CCTV/14_h265_20231216184310.mp4"

model = YOLO("./weights/yolov8l.pt")
BATCH_SIZE = 32
frames = []
cap = cv2.VideoCapture(DSET_FOLDER)
count = 0
while True:
    ret, frame = cap.read()
    frame= cv2.resize(frame, (1080, 720))
    if not ret:
        break
    if(len(frames)!=BATCH_SIZE):
        frames.append(frame)
    else:
        results = model(frames)
        frames = []
        for result in results:
            img = result.orig_img
            classes = result.names
            # boxes = supressDriver(result.boxes)
            for r in result.boxes:
                box = r.xywh.cpu().numpy()[0]
                frame = drawBox(img, box, color=(0, 0, 255))
                id = r.cls
                center = box[:2]
                print(center)
                center = (int(center[0]), int(center[1]))
                pred_class = classes[int(id)]
                cv2.putText(img, f"{pred_class}: {id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            # for r in boxes:
            #     box = r.xywh.cpu().numpy()[0]
            #     frame = drawBox(frame, box)
            #     id = r.cls
            #     center = box[:2]
            #     print(center)
            #     center = (int(center[0]), int(center[1]))
            #     pred_class = classes[int(id)]
            #     cv2.putText(frame,f"{pred_class}: {id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.imshow("frame", img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break