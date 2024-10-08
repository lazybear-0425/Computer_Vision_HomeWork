import cv2
import numpy as np
 
 
img = cv2.imread('1007/example/T_pic.png', cv2.IMREAD_GRAYSCALE)
canny = cv2.Canny(img, 5, 100, None, 3) # 輸出是二值圖
dst = np.copy(canny)
dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

'''這邊要用Polistic的'''
lines_P = cv2.HoughLinesP(canny, 1, np.pi / 180, 33, None, 50, 60)
for line in lines_P:
    line = line[0] # 因為 lines_P 的 shape 是 (9,1,4)
    cv2.line(dst, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv2.LINE_AA)

cv2.imshow('canny', canny)
cv2.imshow('result', dst)
cv2.waitKey(0)

cv2.imwrite('1007/result/hk2/T-canny.png', canny)
cv2.imwrite('1007/result/hk2/T-result.png', dst)
 