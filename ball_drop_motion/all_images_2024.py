# -*- coding: utf-8 -*-
import cv2 as cv
import os

path = os.getcwd()
files = []

# r=root, d=directories, f = files

files = [os.path.join(r, file) for r, d, f in os.walk(path) for file in f if 'topbirak' in file]
#%%
for i, f in enumerate(files):
    print(f"Image {i+1} : {os.path.basename(f)} is being processed.")
    # Read the image
    BGR = cv.imread(f)
    if BGR is None:
        print(f'Could not open the image {f} or find it:', BGR)
        exit(0)
    # Convert to the gray
    imgray = cv.cvtColor(BGR, cv.COLOR_BGR2GRAY) # Convert to Gray
    filtered = cv.medianBlur(imgray, 3)
    # Convert to the Black & White binary image
    _ ,bw = cv.threshold(filtered, 127, 255, cv.THRESH_BINARY)
    
#    cv.imshow('Original image',RGB)
#    cv.imshow('Gray image', imgray)
#    cv.imshow('Threshhold', bw)
    # median filter salt&pepper

#    cv.imshow('Filtered', filtered)
    contours, hierarchy = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for c in contours:

        #calculate moments for each contour
        M = cv.moments(c)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        else:
            cX, cY = 0, 0
    # put text and highlight the center
        cv.circle(BGR, (int(cX), int(cY)), 2, (0, 0, 255), -1)
        cv.putText(BGR, f"{cX:.2f}, {cY:.2f}", (int(cX) + 15, int(cY) ),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    #cv.imshow('Original image',BGR)
    # display the image
    ## Contour Tracing
    # cv.CHAIN_APPROX_SIMPLE for circle returns only one point
    # cv.CHAIN_APPROX_NONE returns all contours
    
    
        cv.drawContours(BGR, contours, -1, (0, 0, 255), 1)
        cv.imshow('Original image', BGR)
        cv.imwrite(f"vid_{os.path.basename(f)}", BGR)
    # =============================================================================
    # for contour in contours:
    #     cv.drawContours(Threshhold, contour, -1, (0, 255, 0), 3)
    #         
    # cnt = contours[0]
    # M = cv.moments(cnt)
    # print(M)
    # =============================================================================
        cv.waitKey(100)

cv.waitKey(0)    
cv.destroyAllWindows()
    
