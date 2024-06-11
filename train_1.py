import os
import cv2
import numpy as np
from PIL import Image

DATASET_PATH = input("Enter Dataset Folder Name:")

recognizer = cv2.face.LBPHFaceRecognizer_create()

def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    eyes = []
    ids = []
    print("Model is Training.\nPlease Wait ...")
    
    for imagePath in imagePaths:
        eye = Image.open(imagePath).convert('L')
        eye_np = np.array(eye)
        
        filename = os.path.splitext(os.path.basename(imagePath))[0]
        parts = filename.split('_')
        id = int(parts[0])
        
        eyes.append(eye_np)
        ids.append(id)
        
        cv2.imshow('Training', eye_np)
        cv2.waitKey(10)
        
    print("Model Trained Successfully")
    return ids, eyes


ids, eyes = getImagesWithID(DATASET_PATH)

recognizer.train(eyes, np.array(ids))
recognizer.save('eye_recognizer.yml')
cv2.destroyAllWindows()
