import cv2

# Load image
image = cv2.imread("input.jpg")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Invert image
inverted_image = 255 - gray_image

# Blur image
blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)

# Invert blurred image
inverted_blurred = 255 - blurred

# Create pencil sketch
sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

# Save output
cv2.imwrite("output_sketch.jpg", sketch)

# Display result
cv2.imshow("Original Image", image)
cv2.imshow("Pencil Sketch", sketch)

cv2.waitKey(0)
cv2.destroyAllWindows()
