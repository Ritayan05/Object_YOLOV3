﻿# Object Detection and Distance Estimation using YOLOv3/YOLOv4 and OpenCV

This project demonstrates real-time object detection and distance estimation using YOLOv3, OpenCV, and Python. The application can detect objects in a video stream (e.g., from a webcam) and estimate their distance from the camera. It also uses text-to-speech functionality to announce the detected objects and their distances.

## Requirements

- Python 3.7
- OpenCV
- NumPy
- PyTesseract
- pyttsx3
- YOLOv3 weights and configuration files
- coco.names (class names for YOLOv3)

## Installation

Install the required packages:

```bash
pip install opencv-python numpy pytesseract pyttsx3
```

Download the YOLOv3 weights and configuration files from the official YOLO website and place them in the project directory.

Download the coco.names file from the official YOLO repository and place it in the project directory.

## Usage

Run the object_detection.py script:

```bash
python main.py
```

Press 'q' to exit the application.

## Code Overview

The code consists of the following main components:

- Video capture: Captures video frames from a webcam using OpenCV.
- Object detection: Detects objects in the video frames using YOLOv3 and OpenCV.
- Distance estimation: Estimates the distance of the detected objects from the camera using the object's bounding box width, focal length, and real-world width.
- Text-to-speech: Announces the detected objects and their distances using the pyttsx3 library.
- Non-Maxima Suppression (NMS): Applies NMS to remove overlapping bounding boxes and improve detection accuracy.

## Configuration

You can configure the following parameters in the code:

- known_width: The real-world width of the object being detected (in centimeters).
- focal_length: The focal length of the camera (experimentally determined or obtained from camera specifications).

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
