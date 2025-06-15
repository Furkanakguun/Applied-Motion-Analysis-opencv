"""
@author: fakgun
"""

import cv2
import os
import numpy as np

# --- Configuration ---
INPUT_FOLDER = "images"
IMAGE_PREFIX = "basket_test_"
IMAGE_EXTENSION = ".jpg"
FRAME_COUNT = 266
ROI_MARGIN_X = 0.2
ROI_MARGIN_Y = 0.2
MIN_AREA = 5
MAX_AREA = 200

# Marker color codes (left to right):
MARKER_COLORS = [
    (0, 0, 255),    # Red - Shoulder
    (0, 255, 0),    # Green - Elbow
    (255, 0, 0),    # Blue - Wrist
    (0, 255, 255)   # Yellow - Middle Finger
]

for i in range(1, FRAME_COUNT + 1):
    filename = f"{IMAGE_PREFIX}{i:03d}{IMAGE_EXTENSION}"
    filepath = os.path.join(INPUT_FOLDER, filename)

    if not os.path.isfile(filepath):
        print(f"Skipping missing file: {filename}")
        continue

    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Define ROI (center region)
    x1 = int(width * ROI_MARGIN_X)
    y1 = int(height * ROI_MARGIN_Y)
    x2 = int(width * (1 - ROI_MARGIN_X))
    y2 = int(height * (1 - ROI_MARGIN_Y))
    roi = gray[y1:y2, x1:x2]

    # Threshold for detecting bright markers
    _, thresh = cv2.threshold(roi, 230, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"]) + x1
        cY = int(M["m01"] / M["m00"]) + y1
        centroids.append((cX, cY))

    # Sort markers from left to right
    centroids = sorted(centroids, key=lambda c: c[0])[:4]

    # Draw lines between consecutive markers (white)
    if len(centroids) == 4:
        for j in range(3):  # draw 3 lines between 4 points
            pt1 = centroids[j]
            pt2 = centroids[j + 1]
            cv2.line(image, pt1, pt2, (255, 255, 255), 2)

    # Draw and label each marker
    for idx, (cX, cY) in enumerate(centroids):
        color = MARKER_COLORS[idx] if idx < len(MARKER_COLORS) else (255, 255, 255)
        cv2.circle(image, (cX, cY), 5, color, -1)
        cv2.putText(image, f"({cX},{cY})", (cX + 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(filepath, image)
