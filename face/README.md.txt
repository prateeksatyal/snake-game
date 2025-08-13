# Facial Gesture Detection

This project detects facial gestures such as blinking, smiling, eyebrow raises, and pimples using OpenCV and dlib.

## Setup

### Download model file

Download the dlib facial landmark predictor model here:

- [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Unzip the downloaded file and put `shape_predictor_68_face_landmarks.dat` in the project folder.

### Install dependencies

```bash
pip install opencv-python dlib numpy
