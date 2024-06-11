import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Function to load the eye dataset
def load_eye_dataset(data_dir):
    images = []
    labels = []

    # Loop through the dataset folder
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Load the image
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Extract the label from the filename
            label = int(file.split("_")[0])

            # Add the image and label to the lists
            images.append(image)
            labels.append(label)

    # Convert the lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Load the eye dataset
dataset_dir = "dataset"
images, labels = load_eye_dataset(dataset_dir)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess the images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape the images
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Convert labels to one-hot encoding
num_classes = len(np.unique(labels))
train_labels = np.eye(num_classes)[train_labels]
test_labels = np.eye(num_classes)[test_labels]

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(56, 56, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, batch_size=32, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Save the model
model.save("eye_model.h5")
