OpenCV Face Detection
This Python script utilizes the OpenCV library to perform real-time face detection using the Haar Cascade Classifier. The Haar Cascade Classifier is a popular object detection algorithm used for identifying objects in images or video frames.

Requirements :-
Python 3.x
OpenCV library

Usage :-

1.Clone the repository:

git clone https://github.com/iayaz/OpenCVFaceDetection.git

2.Navigate to the project directory:

cd OpenCVFaceDetection

3.Install the required dependencies:

pip install opencv-python

4.Run the script:

python face_detection.py

5.Press 'q' to exit the script.

Explanation
face_detection.py: The main Python script that captures video from the default camera, converts frames to grayscale, detects faces using the Haar Cascade Classifier, and displays the video feed with rectangles drawn around detected faces.

haarcascade_frontalface_default.xml: The Haar Cascade Classifier file specifically trained for detecting frontal faces. It is used by OpenCV for face detection.