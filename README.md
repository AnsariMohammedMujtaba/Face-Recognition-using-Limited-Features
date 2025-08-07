# Face Recognition using Limited Features

A Python-based face recognition system that accurately identifies individuals by analyzing limited facial features, specifically focusing on the **eye region**. This approach is particularly useful in scenarios where faces are partially occluded, e.g., with masks.

---

## Features

- **Eye ROI Recognition**: Precise recognition using only the eye region.
- **Real-time Detection**: Works on live webcam feed or video files.
- **Robust Detection**: Uses MediaPipe Face Mesh for accurate eye region extraction.
- **Image Enhancement**: Contrast enhancement and noise reduction applied to eye images.
- **Dataset Management**: Supports multiple users via organized folders.
- **Model Training**: LBPH face recognizer trained on eye region dataset.
- **Recognition with Confidence**: Displays recognized name and confidence scores.
- **Flexible Input**: Choose between webcam or video file input.
- **Cross-platform**: Runs on any system with Python, OpenCV, and MediaPipe installed.

---

## Technologies Used

- Python 3
- OpenCV
- MediaPipe (for face mesh and landmark detection)
- LBPHFaceRecognizer (for eye region recognition)
- NumPy
- PIL (Python Imaging Library)

---

## Installation

1. Clone the repository:

