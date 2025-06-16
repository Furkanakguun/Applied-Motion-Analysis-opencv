"""
@author:fakgun
"""

import cv2
import os
import numpy as np
from scipy.spatial import distance

# --- Configuration ---
INPUT_FOLDER = "images"
IMAGE_PREFIX = "basket_"
IMAGE_EXTENSION = ".jpg"
FRAME_COUNT = 266

ROI_MARGIN_X = 0.2
ROI_MARGIN_Y = 0
MIN_AREA = 5
MAX_AREA = 200
THRESHOLD = 230

# Marker color mapping (shoulder, elbow, wrist, finger)
MARKER_COLORS = [
    (0, 0, 255),    # Red - Shoulder
    (0, 255, 0),    # Green - Elbow
    (255, 0, 0),    # Blue - Wrist
    (0, 255, 255)   # Yellow - Middle Finger
]

previous_markers = []

for i in range(1, FRAME_COUNT + 1):
    filename = f"{IMAGE_PREFIX}{i:03d}{IMAGE_EXTENSION}"
    filepath = os.path.join(INPUT_FOLDER, filename)

    if not os.path.isfile(filepath):
        print(f"Skipping missing file: {filename}")
        continue

    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    x1 = int(width * ROI_MARGIN_X)
    y1 = int(height * ROI_MARGIN_Y)
    x2 = int(width * (1 - ROI_MARGIN_X))
    y2 = int(height * (1 - ROI_MARGIN_Y))
    roi = gray[y1:y2, x1:x2]

    _, thresh = cv2.threshold(roi, THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"]) + x1
        cY = int(M["m01"] / M["m00"]) + y1
        current_centroids.append((cX, cY))

    # Frame 1: use left-to-right X sort to initialize
    if i == 1:
        current_centroids = sorted(current_centroids, key=lambda c: c[0])[:4]
        previous_markers = current_centroids.copy()
    else:
        assigned = []
        new_markers = []
        for prev in previous_markers:
            if not current_centroids:
                break
            dists = [distance.euclidean(prev, c) for c in current_centroids]
            min_index = int(np.argmin(dists))
            new_markers.append(current_centroids[min_index])
            assigned.append(min_index)
        for idx in sorted(set(assigned), reverse=True):
            del current_centroids[idx]
        previous_markers = new_markers + [(0, 0)] * (4 - len(new_markers))

    # Draw tracked markers
    for idx, (cX, cY) in enumerate(previous_markers):
        if cX == 0 and cY == 0:
            continue
        color = MARKER_COLORS[idx]
        cv2.circle(image, (cX, cY), 5, color, -1)
        cv2.putText(image, f"({cX},{cY})", (cX + 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw white polyline connecting markers if all are valid
    if all((x != 0 and y != 0) for (x, y) in previous_markers):
        pts = np.array(previous_markers, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=(255, 255, 255), thickness=2)

    cv2.imwrite(filepath, image)

print("complete")
