import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import time
from ultralytics import YOLO
from dataclasses import dataclass
import math
import logging


def distance(bbox1, bbox2):
    return math.sqrt((bbox1[0] - bbox2[0]) ** 2 + (bbox1[1] - bbox2[1]) ** 2)


def check_line(line, cx, cy):
    v1 = (line[1][0] - line[0][0], line[1][1] - line[0][1])  # Vector 1
    v2 = (line[1][0] - cx, line[1][1] - cy)  # Vector 2
    x = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product (magnitude) Positive--> Below
    return x


def write_counts(img, up_count, down_count):
    up_text = (f"Persons: {up_count[0]} \n"
               f"Cars: {up_count[2]}\n"
               f"Motorcycles: {up_counts[3]}\n"
               f"Buses: {up_count[5]}\n"
               f"Truck: {up_count[7]}")
    y0 = 16
    dy = 25
    for i, line in enumerate(up_text.split('\n')):
        y = y0 + i * dy
        img = cv2.putText(img, line, (480, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (11, 174, 214), thickness=2) #214

    down_text = (f"Persons: {down_count[0]} \n"
                 f"Cars: {down_count[2]}\n"
                 f"Motorcycles: {down_count[3]}\n"
                 f"Buses: {down_count[5]}\n"
                 f"Truck: {down_count[7]}")
    y0 = 150
    dy = 25
    for i, line in enumerate(down_text.split('\n')):
        y = y0 + i * dy
        img = cv2.putText(img, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (11, 174, 214), thickness=2)
    return img


# Load the YOLOv8 model
model = YOLO("yolov8m.pt")

# Open the video file
video_path = r"C:\Users\Chaitanya\Downloads\mohua\MoHUA\video\14_h265_20231216184310.mp4_xxx\sample_video.mp4"
vid_name = video_path.split("/")[-1].split(".")[0]
cap = cv2.VideoCapture(video_path)
person_vehicle_dict = {}
RADIUS = 50
WINDOW_SIZE = 25
drivers = set()
temp_up_list = set()
temp_down_list = set()
down_counts = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 7: 0}
up_counts = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 7: 0}

up_line = ((320, 230), (620, 250))
down_line = ((60, 200), (340, 380))

visited = []

# Set up the logger to log real time counts
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"{vid_name}.log", format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

# Loop through the video frames
logtime = time.time()
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (width//2 , height//2 ))
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0, 1, 2, 3, 5, 7])
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plotting the Up and Down lines
        cv2.line(annotated_frame, up_line[0], up_line[1], color=(255, 0, 0), thickness=2)
        cv2.line(annotated_frame, down_line[0], down_line[1], color=(255, 0, 0), thickness=2)
        annotated_frame = write_counts(annotated_frame, up_counts, down_counts)
        for annotation in zip(classes, boxes, track_ids):
            cls1, box1, id1 = annotation
            if annotation[0] == 0:
                annotated_frame = cv2.circle(annotated_frame, (int(annotation[1][0]), int(annotation[1][1])),
                                             radius=RADIUS, color=(255, 0, 0))
            for annotation2 in zip(classes, boxes, track_ids):
                if annotation2[0] != 0 and annotation2[2] != annotation[2]:
                    id2 = annotation2[2]
                    class2 = annotation2[0]
                    cx2, cy2 = [int(x) for x in annotation2[1][:2]]

                    dist = distance(annotation[1], annotation2[1])
                    if id2 not in temp_up_list and id2 not in temp_down_list and id2 not in visited:
                        # CHECK IF VEHICLE IS BELOW THE UP LINE
                        x_up = check_line(up_line, cx2, cy2)
                        if x_up < 0:
                            temp_up_list.add(id2)
                        else:
                            x_down = check_line(down_line, cx2, cy2)
                            if x_down > 0:
                                temp_down_list.add(id2)
                    if id2 in temp_down_list:
                        x_down = check_line(down_line, cx2, cy2)
                        if x_down < 0:  # If Point has gone below the down line update count
                            down_counts[class2] += 1
                            temp_down_list.remove(id2)
                            visited.append(id2)
                        cv2.drawMarker(annotated_frame, (cx2, cy2), markerType=cv2.MARKER_TRIANGLE_DOWN,
                                       color=(255, 255, 0), thickness=4)
                    if id2 in temp_up_list:
                        x_down = check_line(up_line, cx2, cy2)
                        if x_down > 0:  # If Point has gone below the down line update count
                            up_counts[class2] += 1
                            temp_up_list.remove(id2)
                            visited.append(id2)
                        cv2.drawMarker(annotated_frame, (cx2, cy2), markerType=cv2.MARKER_TRIANGLE_UP,
                                       color=(0, 255, 255), thickness=4)

                    # Log counts
                    if time.time() - logtime > 1:
                        logtime = time.time()
                        logger.info(f"Up Counts: {up_counts}")
                        logger.info(f"Down Counts: {down_counts}")
                    if dist < RADIUS:
                        key = f"{annotation[2]}_{annotation2[2]}"
                        if key not in person_vehicle_dict:
                            person_vehicle_dict[key] = [dist]
                        else:
                            person_vehicle_dict[key].append(dist)
                            if len(person_vehicle_dict[key]) > WINDOW_SIZE:
                                annotated_frame = cv2.line(annotated_frame,
                                                           (int(annotation[1][0]), int(annotation[1][1])),
                                                           (int(annotation2[1][0]), int(annotation2[1][1])),
                                                           color=(0, 0, 255), thickness=4)
                                p_id = key.split("_")[0]
                                drivers.add(p_id)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(person_vehicle_dict)
            print(len(person_vehicle_dict))
            print(f"Drivers {drivers}")
            print(down_counts)
            print(up_counts)
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


