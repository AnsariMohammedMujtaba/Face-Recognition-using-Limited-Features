import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
dataset_path = "F:\Trail\New_1\Eye\dataset"
X = []
y = []
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    for image_path in os.listdir(label_path):
        image = cv2.imread(os.path.join(label_path, image_path), cv2.IMREAD_GRAYSCALE)
        X.append(image.flatten())
        y.append(label)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
model = SVC(kernel='linear', C=1, gamma='auto')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)