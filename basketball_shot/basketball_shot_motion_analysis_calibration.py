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

# Calibration image marker positions (image coordinates)
IMAGE_POINTS = np.array([
    [191, 1014],  # Marker 1 (Bottom Left)
    [149, 900],   # Marker 2
    [147, 780],   # Marker 3
    [144, 496],   # Marker 4 (Top Left)
    [530, 1013],  # Marker 5 (Bottom Right)
    [576, 982],   # Marker 6
    [574, 699],   # Marker 7
    [344, 623],   # Marker 8 (Top Right)
], dtype=np.float32)

# Real-world coordinates for those 8 markers (in cm or units)
WORLD_POINTS = np.array([
    [0, 0],      # Marker 1
    [0, 30],     # Marker 2
    [0, 60],     # Marker 3
    [0, 90],     # Marker 4
    [100, 0],    # Marker 5
    [100, 30],   # Marker 6
    [100, 60],   # Marker 7
    [100, 90],   # Marker 8
], dtype=np.float32)

# Compute homography matrix from image → world coordinates
H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)

# Color for each joint
MARKER_COLORS = [
    (0, 0, 255),    # Shoulder - Red
    (0, 255, 0),    # Elbow - Green
    (255, 0, 0),    # Wrist - Blue
    (0, 255, 255)   # Finger - Yellow
]

previous_markers = []

# --- MAIN FRAME LOOP ---
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
    x2 = int(width * (1 - ROI_MARGIN_X))
    roi = gray[:, x1:x2]

    # Threshold to isolate bright reflective markers
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
        cY = int(M["m01"] / M["m00"])
        current_centroids.append((cX, cY))

    # --- TRACKING MARKERS ---
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

    # --- DRAWING ---
    image_with_markers = image.copy()
    for idx, (cX, cY) in enumerate(previous_markers):
        if cX == 0 and cY == 0:
            continue
        color = MARKER_COLORS[idx]
        cv2.circle(image_with_markers, (cX, cY), 5, color, -1)

        # Project to world coordinates
        pt_img = np.array([[cX, cY]], dtype=np.float32)
        pt_world = cv2.perspectiveTransform(pt_img[None, :, :], H)[0][0]
        wx, wy = pt_world

        cv2.putText(image_with_markers, f"({int(wx)}, {int(wy)})", (cX + 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw line between markers (shoulder to finger)
    if all((x != 0 and y != 0) for (x, y) in previous_markers):
        pts = np.array(previous_markers, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_with_markers, [pts], isClosed=False, color=(255, 255, 255), thickness=2)

    cv2.imwrite(filepath, image_with_markers)

print("complete with calibration.")
