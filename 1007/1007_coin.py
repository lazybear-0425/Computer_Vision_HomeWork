import sys
import cv2
import numpy as np
 
img = cv2.imread('1007/example/coin.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)

rows = blur.shape[0]
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, rows / 8, 
                           param1=100, param2=30, 
                           minRadius=20, maxRadius=100)

for circle in circles[0, :]: # circles.shape : (1, 10, 3)
    circle = np.uint16(np.around(circle))
    center = (circle[0], circle[1])
    cv2.circle(img, center, 1, (0, 100, 200), 3)
    radius = circle[2]
    cv2.circle(img, center, radius, (255, 0, 255), 3)

num_circle = circles.shape[1]
text = f'number of circles: {num_circle}'
cv2.putText(img, text, (img.shape[0] - 60, img.shape[1] - 145), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 100, 0), 1)

cv2.imshow('circle', img)
cv2.waitKey(0)

# 存檔
try:
    import os
    sys.path.append(os.getcwd())
    import mytools # from mytools.py
    mytools.save_file('1007/result/hk3', [['coin-circle.png', img]])
except:
    print('存檔失敗QQ')