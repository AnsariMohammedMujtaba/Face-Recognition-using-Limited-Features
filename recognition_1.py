import cv2
import numpy as np

eye_recognizer = cv2.face.LBPHFaceRecognizer_create()
eye_recognizer.read('eye_recognizer.yml')

cascadePath = cv2.data.haarcascades + "haarcascade_eye.xml"
eyeCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

#names related to ids
names = ['None', 'ABD', 'Asad', 'Mujtaba','CR7', 'ASD', 'Brock Lesner', 'Reigns', 'xyz', 'ZXXXXX',"hassan"]
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
#cam = (r"F:\Trail\New_1\Eye\rono.webp")

#img = cv2.imread(cam)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,225), 1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 50, 225), 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), (50,50,225), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        eye_roi_gray = gray[y:y + h, x:x + w]
        eye_roi_color = img[y:y + h, x:x + w]
        id, confidence = eye_recognizer.predict(eye_roi_gray)
        print(confidence)
        if confidence == 0:
            confidence = "Unknown"
            
        else:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            
            

        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )
        cv2.putText(
            img,
            str(id),
            (x, y - 15),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            1
        )

    cv2.imshow('Eyes Detection', img)
    if cv2.waitKey(1) == 27:
        break

# Do a bit of cleanup
cam.release()
cv2.destroyAllWindows()
