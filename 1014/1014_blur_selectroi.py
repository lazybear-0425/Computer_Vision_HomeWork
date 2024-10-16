import cv2
import numpy as np
import os
import sys

area = ()

def mouse_event(event, x, y, flags, userdata):
    global area
    if event == cv2.EVENT_LBUTTONDOWN:
        area = cv2.selectROI('result', img, False, False)
        x1, x2 = area[0], area[0] + area[2] # 這邊是這樣用的!
        y1, y2 = area[1], area[1] + area[3]
        print(x1, x2, y1, y2)
        print(area)
        roi = np.copy(userdata[y1:y2, x1:x2, :])
        roi = cv2.GaussianBlur(roi, (13, 13), 0)
        userdata[y1:y2, x1:x2, :] = roi
    cv2.imshow('result', userdata)
        

img = cv2.imread('1014/example/people.png')

cv2.imshow('result', img)
cv2.setMouseCallback('result', mouse_event, img)
cv2.waitKey(0)