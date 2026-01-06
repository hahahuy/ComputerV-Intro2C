# Computer Vision - Introduction to Computer Vision

This repository contains my solutions and work for the Computer Vision course assignments. The course focuses on fundamental concepts and practical applications of computer vision.

## Course Overview
This course introduces the basic concepts and techniques of computer vision, including:
- Image processing fundamentals
- Feature detection and matching
- Image segmentation
- Object detection and recognition
- Deep learning applications in computer vision

## Repository Structure
The repository contains the following main components:

### Face Detection and Recognition System
- `face_detection.py`: Core face detection module that implements face detection algorithms using OpenCV and deep learning models. This module provides the fundamental functionality for detecting faces in images and video streams.

- `face_stream_system.py`: Real-time face detection and tracking system that processes video streams from cameras. This module:
  - Captures video input from webcam or video files
  - Processes frames in real-time
  - Detects and tracks faces
  - Displays results with bounding boxes and labels

- `face_stream_verify.py`: Face verification system that:
  - Compares detected faces against a database of known faces
  - Implements face recognition and identity verification
  - Provides real-time feedback on face matches
  - Can be used for access control or identity verification applications

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Other dependencies as specified in each lab's requirements

## Getting Started
1. Clone this repository:
```bash
git clone https://github.com/hahahuy/ComputerV-Intro2C.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the face detection system:
```bash
python face_stream_system.py
```

For face verification:
```bash
python face_stream_verify.py
```

## Author
- Name: Ha Quang Huy
- Student ID: SESEIU21007
- University: HCMIU

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
