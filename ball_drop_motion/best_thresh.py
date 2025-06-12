# -*- coding: utf-8 -*-

import cv2 as cv
from matplotlib import pyplot as plt

RGB = cv.imread('topbirak_041.jpg')
imgray = cv.cvtColor(RGB, cv.COLOR_BGR2GRAY) # Convert to Gray

_ ,thresh1 = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
_ ,thresh2 = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY_INV)
_ ,thresh3 = cv.threshold(imgray, 127, 255, cv.THRESH_TRUNC)
_ ,thresh4 = cv.threshold(imgray, 127, 255, cv.THRESH_TOZERO)
_ ,thresh5 = cv.threshold(imgray, 127, 255, cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [imgray, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

