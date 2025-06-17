import cv2
import os
import numpy as np
from scipy.spatial import distance

# --- CONFIGURATION ---
INPUT_FOLDER = "images"
IMAGE_PREFIX = "basket_"
IMAGE_EXTENSION = ".jpg"
FRAME_COUNT = 266

MIN_AREA = 5
MAX_AREA = 200
THRESHOLD = 230
ROI_MARGIN_X = 0.2

# --- Calibration Data ---

# 8 Marker positions in the image (from kalibrasyon.jpg)
IMAGE_POINTS = np.array([
    [191, 1014],  # Marker 1
    [149, 900],   # Marker 2
    [147, 780],   # Marker 3
    [144, 496],   # Marker 4
    [530, 1013],  # Marker 5
    [576, 982],   # Marker 6
    [574, 699],   # Marker 7
    [344, 623],   # Marker 8
], dtype=np.float32)

# Corresponding real-world coordinates in cm
WORLD_POINTS = np.array([
    [2.5, 30],    # Marker 1
    [2.5, 60],    # Marker 2
    [2.5, 130],   # Marker 3
    [2.5, 190],   # Marker 4
    [108, 10],    # Marker 5
    [108, 80],    # Marker 6
    [108, 120],   # Marker 7
    [108, 180],   # Marker 8
], dtype=np.float32)

# Estimate 2D Conformal Transformation matrix (Affine: scale + rotation + translation)
TRANSFORM_MATRIX, _ = cv2.estimateAffinePartial2D(IMAGE_POINTS, WORLD_POINTS)

# Marker colors: Shoulder, Elbow, Wrist, Middle Finger
MARKER_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255)   # Yellow
]

previous_markers = []

# --- PROCESS FRAMES ---
for i in range(1, FRAME_COUNT + 1):
    filename = f"{IMAGE_PREFIX}{i:03d}{IMAGE_EXTENSION}"
    filepath = os.path.join(INPUT_FOLDER, filename)

    if not os.path.isfile(filepath):
        print(f"Skipping missing file: {filename}")
        continue

    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Apply ROI crop (20% from both sides)
    x1 = int(width * ROI_MARGIN_X)
    x2 = int(width * (1 - ROI_MARGIN_X))
    roi = gray[:, x1:x2]

    # Threshold for detecting bright markers
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
        cX = int(M["m10"] / M["m00"]) + x1  # Offset x due to ROI
        cY = int(M["m01"] / M["m00"])
        current_centroids.append((cX, cY))

    # --- Marker Tracking ---
    if i == 1:
        # Initialize by sorting left to right
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

    # --- Drawing and Coordinate Conversion ---
    annotated = image.copy()
    for idx, (cX, cY) in enumerate(previous_markers):
        if cX == 0 and cY == 0:
            continue
        color = MARKER_COLORS[idx]
        cv2.circle(annotated, (cX, cY), 5, color, -1)

        # Transform to world coordinates using 2D conformal transform
        pt_img = np.array([[cX, cY]], dtype=np.float32)
        pt_world = cv2.transform(np.array([[[cX, cY]]], dtype=np.float32), TRANSFORM_MATRIX)[0][0]
        wx, wy = pt_world

        cv2.putText(annotated, f"({int(wx)},{int(wy)})", (cX + 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw white line connecting markers
    if all((x != 0 and y != 0) for (x, y) in previous_markers):
        pts = np.array(previous_markers, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=False, color=(255, 255, 255), thickness=2)

    cv2.imwrite(filepath, annotated)

print("COMPLETE")
