import cv2
import sys

# Get the path to the input image file
img_path = (r"E:\Project\Eye\CR7.jpg")

# Load the image
img = cv2.imread(img_path)

# Load the face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

# Extract the ROI containing the eyes
for (x, y, w, h) in faces:
    # Extract the region of interest (ROI) containing the face
    roi = img[y:y+h, x:x+w]
    # Convert the ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Detect eyes in the grayscale ROI
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    # Extract the ROI containing the left and right eyes
    for (ex, ey, ew, eh) in eyes:
        if ex < w/2: # left eye
            eye_roi = roi[ey:ey+eh, ex:ex+ew]
            cv2.imshow('Left Eye', eye_roi)
            #cv2.waitKey(0)
        else: # right eye
            eye_roi = roi[ey:ey+eh, ex:ex+ew]
            cv2.imshow('Right Eye', eye_roi)
            cv2.waitKey(0)
