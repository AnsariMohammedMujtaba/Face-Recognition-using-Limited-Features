# Face Recognition using Limited Features

A Python-based face recognition system that accurately identifies individuals by analyzing limited facial features, specifically focusing on the **eye region**. This approach is particularly useful in scenarios where faces are partially occluded, e.g., with masks.

---

## Features

- **Eye ROI Recognition**: Precise recognition using only the eye region
- **Real-time Detection**: Works on live webcam feed or video files
- **Robust Detection**: Uses MediaPipe Face Mesh for accurate eye region extraction
- **Image Enhancement**: Contrast enhancement and noise reduction applied to eye images
- **Dataset Management**: Supports multiple users via organized folders
- **Model Training**: LBPH face recognizer trained on eye region dataset
- **Recognition with Confidence**: Displays recognized name and confidence scores
- **Flexible Input**: Choose between webcam or video file input
- **Cross-platform**: Runs on any system with Python, OpenCV, and MediaPipe installed

---

## Technologies Used

- **Python 3.7+**
- **OpenCV** - Computer vision and image processing
- **MediaPipe** - Face mesh and landmark detection
- **LBPHFaceRecognizer** - Eye region recognition algorithm
- **NumPy** - Numerical operations and array handling
- **PIL/Pillow** - Python Imaging Library for image processing

---

## Requirements

The following Python packages are required to run this project:

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | >=4.5.0 | Computer vision operations |
| `opencv-contrib-python` | >=4.5.0 | Additional OpenCV algorithms (LBPH) |
| `mediapipe` | >=0.8.0 | Face mesh detection |
| `numpy` | >=1.19.0 | Array operations |
| `Pillow` | >=8.0.0 | Image processing |

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Webcam (for real-time detection)

### Step 1: Clone the Repository
