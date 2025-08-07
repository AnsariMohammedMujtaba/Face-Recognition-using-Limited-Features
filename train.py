import os
import cv2
import numpy as np
from PIL import Image

# Base datasets directory (contains subfolders like Name_ID)
DATASETS_PATH = input("Enter base datasets folder (e.g. datasets): ").strip()

recognizer = cv2.face.LBPHFaceRecognizer_create()

def getImagesAndLabels(base_path):
    image_paths = []
    labels = []
    images = []

    print("Model is Training. Please wait...")

    # Loop over each subfolder in the base directory
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Extract numeric ID from folder name: assume format Name_ID
        try:
            person_id = int(folder_name.split("_")[-1])
        except ValueError:
            print(f"Skipping '{folder_name}': invalid ID format.")
            continue

        # Loop over each image file in this person’s folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
                continue

            # Load and convert to grayscale numpy array
            img = Image.open(file_path).convert("L")
            img_np = np.array(img)

            images.append(img_np)
            labels.append(person_id)

            # Optional: display the eye during training
            cv2.imshow("Training", img_np)
            cv2.waitKey(10)

    cv2.destroyAllWindows()
    print("Model training data collection complete.")
    return images, labels

# Collect images and labels from subfolders
images, labels = getImagesAndLabels(DATASETS_PATH)

# Train and save the LBPH recognizer
recognizer.train(images, np.array(labels))
recognizer.save("eye_recognizer.yml")
print("Model trained and saved as 'eye_recognizer.yml'.")
