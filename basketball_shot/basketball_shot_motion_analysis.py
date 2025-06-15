import cv2
import numpy as np
import os


def find_centroids(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to isolate bright markers
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find contours of the markers
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

    # Optional: sort left to right by x
    centroids = sorted(centroids, key=lambda x: x[0])

    return centroids


# Example usage over a directory of images
image_dir = "images"
results = {}

for filename in sorted(os.listdir(image_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(image_dir, filename)
        centroids = find_centroids(path)
        results[filename] = centroids
        print(f"{filename}: {centroids}")
