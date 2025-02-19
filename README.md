# Vehicle Detection and Counting using YOLOv8

This project is designed to detect and count different types of vehicles (such as cars, motorcycles, buses, and trucks) crossing predefined lines in a video. The project uses the YOLOv8 model for object detection and tracking, and it counts the number of vehicles crossing the "up" and "down" lines.In case of motorcycle the rider bounding box is appended with the motorcycle bounding box so that they are considered onr body.

## Introduction

The goal of this project is to detect vehicles in a video stream and count how many of them cross two predefined lines (referred to as the "up" and "down" lines). The project uses the YOLOv8 model for object detection and tracking, and it provides real-time counts of vehicles crossing these lines.

## Features

- **Vehicle Detection**: Detects vehicles such as cars, motorcycles, buses, and trucks using the YOLOv8 model.
- **Line Crossing Detection**: Tracks vehicles and counts how many cross the predefined "up" and "down" lines.
- **Real-Time Counting**: Provides real-time counts of vehicles crossing the lines.
- **Visualization**: Visualizes the detected vehicles, the lines, and the counts on the video frame.
- **Logging**: Logs the counts of vehicles crossing the lines at regular intervals.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/vehicle-detection-counting.git
   cd vehicle-detection-counting
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7 or later installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 Model**:
   The YOLOv8 model weights (`yolov8m.pt`) will be automatically downloaded when you run the script for the first time. Alternatively, you can manually download it from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/ultralytics).

## Usage

1. **Prepare the Video**:
   Place your video file in the appropriate directory and update the `video_path` variable in the script to point to your video file.

2. **Run the Script**:
   Run the script to start the vehicle detection and counting process:
   ```bash
   python vehicle_detection.py
   ```

3. **View the Results**:
   The script will display the video with the detected vehicles, the lines, and the counts. The counts will also be logged in a file named after the video.

4. **Stop the Script**:
   Press `q` to stop the script and view the final counts and other statistics.

## Results

- **Real-Time Visualization**: The script displays the video with bounding boxes around detected vehicles, the lines, and the counts of vehicles crossing the lines.
- **Logging**: The counts of vehicles crossing the lines are logged in a file for further analysis.
- **Final Statistics**: When the script is stopped, it prints the final counts of vehicles crossing the lines and other statistics.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

1. **Fork the Repository**:
   Fork the repository to your own GitHub account.

2. **Create a Branch**:
   Create a branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**:
   Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add your commit message here"
   ```

4. **Push to the Branch**:
   Push your changes to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**:
   Open a pull request to the main repository.
