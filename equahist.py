import cv2
import numpy as np

# Assuming you have the eye image stored in the 'eye_image' variable
img_path = (r"F:\Trail\New_1\Eye\d1\7_8_right_eye_100.jpg")

# Load the image
img = cv2.imread(img_path)

# Convert the eye image to grayscale
gray_eye_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform histogram equalization
equalized_eye_image = cv2.equalizeHist(gray_eye_image)

# Display the equalized eye image
cv2.imshow('Equalized Eye Image', equalized_eye_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
