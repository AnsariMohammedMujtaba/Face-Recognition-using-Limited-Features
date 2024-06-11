import cv2
import os

# Load the face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create a folder to store the eye images
dir_name = input("Enter Directory name:")
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Input ID and name
id = input('Enter ID: ')
name = input('Enter Name: ')

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
        roi = gray[y:y+h, x:x+w]

        # Detect eyes in the grayscale ROI
        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5)

        # Extract the ROI containing the left and right eyes
        for (ex, ey, ew, eh) in eyes:
            if ex < w/2: # left eye
                eye_roi = roi[ey:ey+eh, ex:ex+ew]
                # Resize the left eye image to a fixed size (e.g., 100x100)
                eye_roi = cv2.resize(eye_roi, (56, 56))
                # Display the left eye image
                cv2.imshow('left_eye', eye_roi)
                # Create a filename for the left eye image
                filename = f'{dir_name}/{name}_{id}_left_eye_{i}_{ex}.jpg'
                # Save the left eye image
                cv2.imwrite(filename, eye_roi)
            else: # right eye
                eye_roi = roi[ey:ey+eh, ex:ex+ew]
                # Resize the right eye image to a fixed size (e.g., 100x100)
                eye_roi = cv2.resize(eye_roi, (56, 56))
                # Display the right eye image
                cv2.imshow('right_eye', eye_roi)
                # Create a filename for the right eye image
                filename = f'{dir_name}/{name}_{id}_right_eye_{i}_{ex}.jpg'
                # Save the right eye image
                cv2.imwrite(filename, eye_roi)

    # Display the frame
    cv2.imshow('frame',frame)

    # Wait for 1 second between each frame capture
    cv2.waitKey(1000)

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
