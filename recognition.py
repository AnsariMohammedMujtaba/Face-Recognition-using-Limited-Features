import cv2
import numpy as np

CLASSIFIER_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
classifier = cv2.CascadeClassifier(CLASSIFIER_PATH)

eye_recognizer = cv2.face.LBPHFaceRecognizer_create()
eye_recognizer.read('eye_recognizer.yml')

cam = cv2.VideoCapture(0)
id = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        eye_roi_gray = gray[y:y+h, x:x+h]
        eye_roi_color = frame[y:y+h, x:x+h]
        id, conf = eye_recognizer.predict(gray[y:y+h, x:x+w])

        if conf < 50:
            name = str(id)
            frame = cv2.putText(frame, name, (x, y-10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            print("Recognized Person: ", name)
            
    cv2.imshow('Eyes Recognition', frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
