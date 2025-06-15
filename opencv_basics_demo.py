# opencv_basics_demo.py

import cv2

# 1. Reading and Displaying an Image
image = cv2.imread('basketball.jpg')  # Load an image from file
cv2.imshow('Original Image', image)  # Display the original image
cv2.waitKey(0)

# 2. Converting to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)

# 3. Resizing and Cropping
resized = cv2.resize(image, (640, 480))  # Resize image to 640x480
cv2.imshow('Resized Image', resized)
cv2.waitKey(0)

cropped = image[100:400, 200:600]  # Crop the image using slicing
cv2.imshow('Cropped Image', cropped)
cv2.waitKey(0)

# 4. Drawing Shapes
# Draw a rectangle: (image, start_point, end_point, color, thickness)
cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), 2)

# Draw a circle: (image, center_coordinates, radius, color, thickness)
cv2.circle(image, (150, 150), 50, (255, 0, 0), 3)

cv2.imshow('Shapes Drawn', image)
cv2.waitKey(0)

# 5. Edge Detection (Canny)
edges = cv2.Canny(gray, 100, 200)  # Detect edges using the Canny method
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)

# 6. Reading from Video or Webcam
# cap = cv2.VideoCapture(0)  # Uncomment to use webcam (0 is the default camera index)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow('Webcam Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()

# 7. Contour Detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 255), 2)  # Draw all contours
cv2.imshow('Contours', image)
cv2.waitKey(0)

cv2.destroyAllWindows()
