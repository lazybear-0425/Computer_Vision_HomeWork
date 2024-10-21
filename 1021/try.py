import numpy as np
import cv2

img = cv2.imread('1021/example/block.jpg')
h = 48
w = 48
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
_, thres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cord = [[912, 163], [1018, 266], [1012, 163], [915, 266]]; cord.sort()
origin_cord = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)
my_cord = np.array(cord, dtype=np.float32)

M = cv2.getPerspectiveTransform(my_cord, origin_cord)
# M2 = cv2.findHomography(my_cord, origin_cord)

persp = cv2.warpPerspective(thres, M, (w, h))

one_or_zero = np.zeros((6, 6))
for i in range(w):
    for j in range(h):
        if persp[i][j] == 0:
            one_or_zero[(i // 8)][(j // 8)] += 1
one_or_zero = one_or_zero / 64

num = 0
for i in range(1, 5):
    for j in range(1, 5):
        num *= 2
        if one_or_zero[i][j] >= 0.5: num += 1
        else: num += 0
print(num)
print(f'{num:b}')

cv2.imshow('result', persp)
cv2.waitKey(0)