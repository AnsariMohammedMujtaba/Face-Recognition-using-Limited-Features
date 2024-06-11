import cv2
import numpy as np

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the eye detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Train the eye detector using LBPH (optional)
# eye_recognizer = cv2.face.LBPHFaceRecognizer_create()
# eye_recognizer.read('eye_recognizer.xml')

# Initialize the video capture object
cap = cv2.VideoCapture(0)
id= input('ID: ')
num = 0

# Start capturing and processing the frames
while True:
    
    # Read a frame from the video capture object
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Detect the eyes
    for (x, y, w, h) in faces:
        num += 1
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]
            cv2.imwrite('dataset/'+str(id)+'.'+str(num)+'.jpg', gray[ey:ey+eh, ex:ex+ew])
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
            # Preprocess the eye_roi_gray and eye_roi_color for further analysis

    # Display the processed frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if (num>30):
         break
    # Wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break
