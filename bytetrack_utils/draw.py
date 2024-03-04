import numpy as np
import cv2

# Define the dimensions of the image
image_width = 400
image_height = 300

# Create a black background image
black_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Define the coordinates of the top-left and bottom-right corners of the green box
top_left = (100, 100)
bottom_right = (300, 200)

# Draw a green box on the black background image
cv2.rectangle(black_image, top_left, bottom_right, (0, 255, 0), thickness=2)

# Display the image
cv2.imshow('Green Box on Black Background', black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()