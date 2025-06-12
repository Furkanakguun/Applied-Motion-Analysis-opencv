# -*- coding: utf-8 -*-
import cv2 as cv

BGR = cv.imread('moment.jpg')
imgray = cv.cvtColor(BGR, cv.COLOR_BGR2GRAY) # Convert to Gray

_ ,thresh1 = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)

M = cv.moments(thresh1)
# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv.circle(BGR, (cX, cY), 5, (0, 0, 255), -1)
cv.putText(BGR, "centroid", (cX - 15, cY - 15),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv.imshow('Original image', BGR)
cv.waitKey(0)