# Vehicle Detection and Counting with YOLOv8

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-111F68)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)

A computer-vision pipeline that counts traffic in CCTV video, **by vehicle type and by direction of travel**.
Each frame goes through YOLOv8 for detection, YOLOv8's built-in tracker keeps a persistent ID on every object
across frames, and a vehicle is counted when its centre crosses one of two counting lines drawn across the
road. Running counts are drawn on the video and written to a log file every second.

A sample clip is included in the repository, so the pipeline can be run without downloading a dataset.

---

## What it does

- **Detects six classes**: person, bicycle, car, motorcycle, bus and truck (COCO classes `0, 1, 2, 3, 5, 7`).
  All other COCO classes are filtered out at inference time. Vehicles are counted; people are detected only
  for rider pairing and are not counted.
- **Tracks objects across frames** with YOLOv8's built-in tracker (`model.track(..., persist=True)`), so each
  vehicle keeps the same ID from frame to frame.
- **Counts by direction**: an "up" line and a "down" line, one for each side of the road. A vehicle is counted
  when its centre moves from one side of a line to the other.
- **Counts each vehicle once**: an ID that has already been counted is remembered and never counted again.
- **Shows live results**: bounding boxes, track IDs, both counting lines, per-class counts for each direction,
  and a triangle marker on each vehicle that is about to cross.
- **Logs counts every second** to a `.log` file named after the video, for later analysis.
- **Links likely riders to vehicles (experimental)**: two tracked objects that stay within 50 px of each other
  for more than 25 frames are joined on screen with a red line as a likely rider and vehicle.

---

## How the counting works

1. **Read** – OpenCV reads the video frame by frame and resizes each frame to half its size.
2. **Detect and track** – YOLOv8 returns a box, a class and a confidence score for each object, restricted to
   the six classes above; the tracker attaches a persistent ID to each box.
3. **Assign a side** – for each counting line, the 2D cross product of the line's direction vector and the
   vector to the vehicle's centre gives its sign: which side of the line the vehicle is on.
4. **Count the crossing** – when that sign flips, the vehicle has crossed. The count for its class and
   direction goes up by one, and its ID is added to a `visited` list so it is not counted twice.

The two lines are defined in pixel coordinates on the half-size frame and are tuned to the sample camera:

```python
up_line   = ((320, 230), (620, 250))
down_line = ((60, 200), (340, 380))
```

---

## Getting started

### Requirements

- Python 3.9 or later
- `ultralytics`, `opencv-python` and `lap` (the tracker needs `lap` for track association)

### Install

```bash
git clone https://github.com/codedchaitanya/Vehicle_dectection_and_counting.git
cd Vehicle_dectection_and_counting
pip install ultralytics opencv-python lap
```

### Run on the sample video

The working entry point is `Major/track.py`. Point `video_path` near the top of the script at the included
clip, using forward slashes:

```python
video_path = "video/14_h265_20231216184310.mp4_xxx/sample_video.mp4"
```

Then run it from the `Major` folder:

```bash
cd Major
python track.py
```

A window opens with the annotated video and live counts. Counts are logged every second to
`sample_video.log`. Press **`q`** to stop; the final up and down counts per class are printed in the terminal.

The script loads `yolov8m.pt`. If the file is not in the working folder, Ultralytics downloads it automatically
on the first run.

### Using your own footage

Change `video_path` to your video, then adjust `up_line` and `down_line` to fit the new camera angle.
The coordinates apply to the half-size frame. Lines that do not cross the road in your footage will count nothing.

---

## Project structure

```
Major/
├── track.py                   # Main pipeline: detection, tracking, line-crossing counts, logging
├── video/                     # Sample CCTV clip
├── weights/                   # YOLOv8 weights (yolov8m, yolov8l)
├── templates/                 # Digit templates (0-9) for reading the CCTV timestamp
├── helper.py                  # Bounding-box conversions, annotation reader, class colour map
│
│   Development and experiment scripts
├── vehicle_detection.py       # Batch detection; IoU-based rider suppression (not wired in yet)
├── infer_vid.py               # Exports frames and YOLO-format annotations from a video
├── counting.py                # Draws the exported annotations on a frame for checking
├── dataset_preprocess.ipynb   # Class distribution of the Dhaka traffic dataset (imbalance check)
├── train.py                   # Fine-tunes YOLOv8n on the Dhaka traffic dataset (10 epochs)
├── val.py                     # Evaluates the fine-tuned model: mAP50, mAP75, mAP50-95, per-class mAP
└── template_matching.py       # Reads the on-screen CCTV date and time with pytesseract OCR
```

The development scripts expect datasets and trained weights that are not part of this repository
(`../datasets/Patiala CCTV/`, `../datasets/dhaka_traffic_dataset/`, `runs/detect/train4/`). They are kept as a
record of the experiments: fine-tuning on South Asian traffic, which COCO does not cover well, and OCR of the
CCTV timestamp.

---

## Tech stack

| Tool | Role |
|---|---|
| **YOLOv8 (Ultralytics)** | Object detection, built-in multi-object tracking, fine-tuning and evaluation |
| **OpenCV** | Video input, resizing, drawing boxes, lines, markers and counts |
| **pytesseract** | OCR of the timestamp printed on the CCTV footage |
| **NumPy, Matplotlib** | Annotation processing and dataset class-distribution plots |

---

## Known limitations and next steps

- **Rider pairing is experimental.** The pairing check is not yet limited to the person class, and a paired
  rider and motorcycle are not yet merged into one object. The links are drawn on screen only and do not change
  the counts.
- **Frames with nothing to track stop the run.** When no object is tracked, `boxes.id` is `None`; those frames
  should be skipped.
- **A vehicle alone in the frame is not counted.** The crossing check runs inside a loop over pairs of objects,
  so it needs at least two objects on screen. Moving the check out of the pair loop fixes this and also makes
  each frame cheaper.
- **Counting lines are fixed** to the sample camera angle and need new coordinates for other footage.
- **No ground-truth check yet.** The counts have not been compared with a hand-counted clip, so no accuracy
  figure is claimed.
- **The fine-tuned Dhaka model is separate from the counter.** `track.py` uses the standard COCO-trained
  YOLOv8m; the next step is to plug the fine-tuned model in for local vehicle types such as auto-rickshaws.
