import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CALIBRATION_IMAGE = "kalibrasyon.jpg"
MIN_AREA = 5
MAX_AREA = 500
THRESHOLD = 230

# --- LOAD & THRESHOLD IMAGE ---
image = cv2.imread(CALIBRATION_IMAGE)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)

# --- FIND CONTOURS ---
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- FILTER AND CALCULATE CENTROIDS ---
centroids = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA or area > MAX_AREA:
        continue
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroids.append((cX, cY))

# --- SORT INTO LEFT/RIGHT GROUPS ---
centroids_sorted = sorted(centroids, key=lambda c: c[0])
mid_x = (centroids_sorted[0][0] + centroids_sorted[-1][0]) // 2

left_col = sorted([pt for pt in centroids if pt[0] < mid_x], key=lambda c: -c[1])[:4]
right_col = sorted([pt for pt in centroids if pt[0] >= mid_x], key=lambda c: -c[1])[:4]

# --- COMBINE AND RETURN FINAL ORDER ---
image_points = np.array(left_col + right_col, dtype=np.float32)

# --- DRAW FOR VERIFICATION ---
annotated = image.copy()
for idx, (x, y) in enumerate(image_points):
    cx, cy = int(x), int(y)
    cv2.circle(annotated, (cx, cy), 6, (0, 255, 255), -1)
    cv2.putText(annotated, f"{idx+1}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# --- SHOW IMAGE WITH LABELS ---
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 10))
plt.imshow(annotated_rgb)
plt.title("Calibration Markers: Labeled 1–8")
plt.axis("off")
plt.show()

# --- OUTPUT RESULTS ---
print("IMAGE_POINTS = np.array([")
for (x, y) in image_points:
    print(f"    [{x}, {y}],")
print("], dtype=np.float32)")
