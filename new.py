import cv2
import os
import sys

# Capture video from camera
cap = cv2.VideoCapture(0)

id = input('ID: ')
name = input('Name: ')
num = 0

# Collect 30 images dataset of the same person
for i in range(30):
    # Capture frame-by-frame
    ret, frame = cap.read()
# # Continuously capture frames from the camera
# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the face and eye detection models
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Extract the ROI containing the eyes
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        roi = gray[y:y+h, x:x+w]
        # Detect eyes in the grayscale ROI
        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5)

        # Extract the ROI containing the left and right eyes
        for (ex, ey, ew, eh) in eyes:
            if ex < w/2: # left eye
                eye_roi = roi[ey:ey+eh, ex:ex+ew]
                # Display the left eye image
                cv2.imshow('left_eye', eye_roi)
                # Create a filename for the left eye image
                filename = 'dataset/left_eye_'+str(name)+"."+str(id)+"."+str(num)+".jpg".format(i, ex)
                # Save the left eye image
                cv2.imwrite(filename , eye_roi)
            else: # right eye
                eye_roi = roi[ey:ey+eh, ex:ex+ew]
                # Display the right eye image
                cv2.imshow('right_eye', eye_roi)
                # Create a filename for the right eye image
                filename = 'dataset/right_eye_'+str(name)+"."+str(id)+"."+str(num)+".jpg".format(i, ex)
                # Save the right eye image
                cv2.imwrite(filename, eye_roi)
                
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
