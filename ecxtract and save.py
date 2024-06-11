import cv2
import sys
import os

# Get the path to the input image file
img_path = (r"D:\FaceRecog-main\FaceRecog-main\rono.webp")

# Load the image
img = cv2.imread(img_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Load the face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)       

# Create a folder to store the eye images
if not os.path.exists("dataset"):
    os.makedirs("dataset")

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
            eye_roi = cv2.resize(eye_roi, (56, 56))
            # Create a filename for the left eye image
            filename = os.path.splitext(os.path.basename(img_path))[0] + '_left_eye.jpg'
            # Save the left eye image
            cv2.imwrite ('dataset/' +filename, eye_roi)
        else: # right eyda            
            eye_roi = roi[ey:ey+eh, ex:ex+ew]
            # Create a filename for the right eye image
            filename = os.path.splitext(os.path.basename(img_path))[0] + '_right_eye.jpg'
            # Save the right eye image
            cv2.imwrite('dataset/' +filename, eye_roi)

# for (x, y, w, h) in faces:
#     # Extract the region of interest (ROI) containing the face
#     roi = img[y:y+h, x:x+w]
#     # Convert the ROI to grayscale
#     roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     # Detect eyes in the grayscale ROI
#     eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
#     # Extract the ROI containing the left and right eyes
#     for (ex, ey, ew, eh) in eyes:
#         if ex < w/2: # left eye
#             eye_roi = roi[ey:ey+eh, ex:ex+ew]
#             filename = os.path.splitext(os.path.basename(img_path))[0] + '_left_eye.jpg'
#             # Save the left eye image
#             cv2.imwrite("dataset/" +filename, eye_roi)
#         else: # right eye
#             eye_roi = roi[ey:ey+eh, ex:ex+ew]
#             filename = os.path.splitext(os.path.basename(img_path))[0] + '_right_eye.jpg'
#             # Save the right eye image
#             cv2.imwrite("dataset/"+ filename, eye_roi)