import cv2
import numpy as np
import os
import sys

last_x = -1
last_y = -1

def mouse_event(event, x, y, flags, userdata):
    global last_x
    global last_y
    if event == cv2.EVENT_LBUTTONDOWN:    
        last_x = x
        last_y = y
    elif event == cv2.EVENT_MOUSEMOVE and last_x != -1 and last_y != -1:
        userdata = np.copy(userdata)
        cv2.rectangle(userdata, (last_x, last_y), (x, y), (0, 0, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        # cv2.rectangle(userdata, (last_x, last_y), (x, y), (0, 0, 255), 2)
        roi = np.copy(userdata[last_y:y, last_x:x, :])
        roi = cv2.GaussianBlur(roi, (13, 13), 0)
        userdata[last_y:y, last_x:x, :] = roi
        # init
        last_x = -1
        last_y = -1
    cv2.imshow('result', userdata)
        

img = cv2.imread('1014/example/people.png')

cv2.imshow('result', img)
cv2.setMouseCallback('result', mouse_event, img)
cv2.waitKey(0)

try:
    sys.path.append(os.getcwd())
    import mytools
    mytools.save_file('1014/result/hk1', [['blur+roi.png', img]])
except Exception as e:
    print('存檔失敗QQ，原因:', e)