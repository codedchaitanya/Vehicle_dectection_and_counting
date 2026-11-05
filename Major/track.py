"""
Vehicle detection, tracking and line-crossing counts with YOLOv8.

Usage (from any folder):
    python Major/track.py                         # sample clip, yolov8m, live window
    python Major/track.py --video path/to/clip.mp4
    python Major/track.py --weights yolov8n.pt    # smaller, faster model
    python Major/track.py --save output.mp4       # also save the annotated video
    python Major/track.py --no-display            # no window (servers, CI); counts still logged

Press 'q' in the window to stop early. Final counts are printed either way.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import logging
import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Paths are resolved relative to this file, so the script runs from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "video" / "14_h265_20231216184310.mp4_xxx" / "sample_video.mp4"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# COCO class ids used by the pipeline
PERSON = 0
VEHICLE_CLASSES = {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
TRACKED_CLASSES = [PERSON] + list(VEHICLE_CLASSES)
PLURALS = {1: "Bicycles", 2: "Cars", 3: "Motorcycles", 5: "Buses", 7: "Trucks"}

# Counting lines, in pixels on the half-size frame. Tuned to the sample camera.
UP_LINE = ((320, 230), (620, 250))
DOWN_LINE = ((60, 200), (340, 380))

# Rider pairing: a person within RADIUS px of the same vehicle for more than WINDOW_SIZE frames
RADIUS = 50
WINDOW_SIZE = 25

LOG_INTERVAL_S = 1.0
TEXT_COLOUR = (11, 174, 214)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def check_line(line, cx, cy):
    """Signed 2D cross product: its sign tells which side of the line (cx, cy) is on."""
    v1 = (line[1][0] - line[0][0], line[1][1] - line[0][1])
    v2 = (line[1][0] - cx, line[1][1] - cy)
    return v1[0] * v2[1] - v1[1] * v2[0]


def write_counts(img, up_counts, down_counts):
    """Draw the running counts: UP block top right, DOWN block on the left."""
    for counts, x, y0 in ((up_counts, 480, 16), (down_counts, 15, 150)):
        for i, (cls_id, name) in enumerate(PLURALS.items()):
            cv2.putText(img, f"{name}: {counts[cls_id]}", (x, y0 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOUR, thickness=2)
    return img


class LineCounter:
    """Counts each tracked vehicle once, when its centre crosses the UP or DOWN line."""

    def __init__(self, up_line, down_line):
        self.up_line = up_line
        self.down_line = down_line
        self.waiting_up = set()     # ids approaching the UP line, not yet across
        self.waiting_down = set()   # ids approaching the DOWN line, not yet across
        self.counted = set()        # ids already counted: never counted twice
        self.up_counts = {c: 0 for c in VEHICLE_CLASSES}
        self.down_counts = {c: 0 for c in VEHICLE_CLASSES}

    def update(self, track_id, cls_id, cx, cy, frame):
        if track_id in self.counted:
            return
        # First sighting: note which line the vehicle is approaching
        if track_id not in self.waiting_up and track_id not in self.waiting_down:
            if check_line(self.up_line, cx, cy) < 0:
                self.waiting_up.add(track_id)
            elif check_line(self.down_line, cx, cy) > 0:
                self.waiting_down.add(track_id)

        if track_id in self.waiting_down:
            if check_line(self.down_line, cx, cy) < 0:      # crossed the DOWN line
                self.down_counts[cls_id] += 1
                self.waiting_down.discard(track_id)
                self.counted.add(track_id)
            cv2.drawMarker(frame, (cx, cy), markerType=cv2.MARKER_TRIANGLE_DOWN,
                           color=(255, 255, 0), thickness=4)

        if track_id in self.waiting_up:
            if check_line(self.up_line, cx, cy) > 0:        # crossed the UP line
                self.up_counts[cls_id] += 1
                self.waiting_up.discard(track_id)
                self.counted.add(track_id)
            cv2.drawMarker(frame, (cx, cy), markerType=cv2.MARKER_TRIANGLE_UP,
                           color=(0, 255, 255), thickness=4)


class RiderPairing:
    """Flags a person who stays close to the same vehicle for many frames as its likely rider."""

    def __init__(self, radius, window):
        self.radius = radius
        self.window = window
        self.close_frames = {}      # (person_id, vehicle_id) -> frames seen within radius
        self.riders = set()         # (person_id, vehicle_id) pairs flagged as rider + vehicle

    def update(self, persons, vehicles, frame):
        for pid, (px, py) in persons:
            cv2.circle(frame, (px, py), radius=self.radius, color=(255, 0, 0))
            for vid, (vx, vy) in vehicles:
                if distance((px, py), (vx, vy)) >= self.radius:
                    continue
                key = (pid, vid)
                self.close_frames[key] = self.close_frames.get(key, 0) + 1
                if self.close_frames[key] > self.window:
                    self.riders.add(key)
                    cv2.line(frame, (px, py), (vx, vy), color=(0, 0, 255), thickness=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Count vehicles crossing two lines in a traffic video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="input video (default: the sample clip)")
    parser.add_argument("--weights", default="yolov8m.pt",
                        help="YOLOv8 weights: a file name (downloaded into Major/weights/ if missing) or a path")
    parser.add_argument("--save", type=Path, help="also write the annotated video to this .mp4 file")
    parser.add_argument("--no-display", action="store_true", help="run without opening a window")
    return parser.parse_args()


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute() and weights.parent == Path("."):
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        weights = WEIGHTS_DIR / weights.name           # Ultralytics downloads official weights here if missing
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: could not open video file: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    log_path = PROJECT_ROOT / f"{args.video.stem}.log"
    logging.basicConfig(filename=log_path, format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    counter = LineCounter(UP_LINE, DOWN_LINE)
    pairing = RiderPairing(RADIUS, WINDOW_SIZE)
    writer = None
    frames = 0
    last_log = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break
        frames += 1
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 2, height // 2))

        # Track people and vehicles, keeping track ids between frames
        result = model.track(frame, persist=True, classes=TRACKED_CLASSES, verbose=False)[0]
        annotated = result.plot()
        cv2.line(annotated, UP_LINE[0], UP_LINE[1], color=(255, 0, 0), thickness=2)
        cv2.line(annotated, DOWN_LINE[0], DOWN_LINE[1], color=(255, 0, 0), thickness=2)

        # A frame with nothing tracked has boxes.id = None: show it and move on
        if result.boxes.id is not None:
            centres = result.boxes.xywh.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            classes = [int(c) for c in result.boxes.cls.tolist()]

            persons, vehicles = [], []
            for cls_id, (cx, cy, _, _), tid in zip(classes, centres, track_ids):
                centre = (int(cx), int(cy))
                if cls_id == PERSON:
                    persons.append((tid, centre))
                else:
                    vehicles.append((tid, centre))
                    counter.update(tid, cls_id, *centre, annotated)   # every vehicle, even alone in frame
            pairing.update(persons, vehicles, annotated)

        annotated = write_counts(annotated, counter.up_counts, counter.down_counts)

        if time.time() - last_log >= LOG_INTERVAL_S:
            last_log = time.time()
            logger.info(f"Up Counts: {counter.up_counts}")
            logger.info(f"Down Counts: {counter.down_counts}")

        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("YOLOv8 Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    logger.info(f"Final Up Counts: {counter.up_counts}")
    logger.info(f"Final Down Counts: {counter.down_counts}")
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Summary, whether the video ended or 'q' was pressed
    print("\n" + "=" * 36)
    print(f"FINAL COUNTS  ({frames} frames, {weights.name})")
    print("=" * 36)
    print(f"{'Class':<12}{'UP':>6}{'DOWN':>8}")
    for cls_id, name in VEHICLE_CLASSES.items():
        print(f"{name:<12}{counter.up_counts[cls_id]:>6}{counter.down_counts[cls_id]:>8}")
    print(f"{'Total':<12}{sum(counter.up_counts.values()):>6}{sum(counter.down_counts.values()):>8}")
    print(f"\nLikely riders (person + vehicle pairs): {len(pairing.riders)}")
    print(f"Per-second counts logged to: {log_path}")
    if args.save:
        print(f"Annotated video saved to: {args.save}")


if __name__ == "__main__":
    main()
