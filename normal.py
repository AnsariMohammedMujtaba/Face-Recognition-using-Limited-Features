import cv2
import numpy as np

# Assuming you have the eye image stored in the 'eye_image' variable
img_path = (r"F:\Trail\New_1\Eye\abd.jpg")

# Load the image
img = cv2.imread(img_path)

# Convert the eye image to grayscale
gray_eye_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Normalize the grayscale image
normalized_eye_image = cv2.equalizeHist(gray_eye_image)

# Perform Gaussian blur on the normalized image
blurred_eye_image = cv2.GaussianBlur(normalized_eye_image, (5, 5), 0)

# Normalize the blurred image between 0 and 1
normalized_blurred_eye_image = blurred_eye_image / 255.0

# Display the normalized eye image
cv2.imshow('Normalized Eye Image', normalized_blurred_eye_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
