import cv2
import sys
import os

# Load the face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create a folder to store the eye images
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Collect 30 images dataset of the same person
for i in range(30):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    #Extract the ROI containing the eyes
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the grayscale ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        # Extract the ROI containing the left and right eyes
        for (ex, ey, ew, eh) in eyes:
            if ex < w/2: # left eye
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                # Display the left eye image
                cv2.imshow('left_eye', eye_roi)
                # Create a filename for the left eye image
                filename = 'dataset/{}_left_eye_{}.jpg'.format(i, ex)
                # Save the left eye image
                cv2.imwrite(filename, eye_roi)
                # Draw a rectangle around the left eye
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
            else: # right eye
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                # Display the right eye image
                cv2.imshow('right_eye', eye_roi)
                # Create a filename for the right eye image
                filename = 'dataset/{}_right_eye_{}.jpg'.format(i, ex)
                # Save the right eye image
                cv2.imwrite(filename, eye_roi)
                # Draw a rectangle around the right eye
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Wait for 1 second between each frame capture
    cv2.waitKey(1000)

# When everything done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
