import numpy as np
import cv2

img = cv2.imread('1021/example/block.jpg')
h = 48
w = 48
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
_, thres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thres = cv2.erode(thres, kernel)
# 給特定那個區塊用的
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# thres = cv2.erode(thres, kernel)

cord = np.array([[1224, 796], [1126, 903], [1227, 897], [1128, 802]], dtype=np.float32)
max = np.argmax(np.sum(cord, axis = 1))     
cord[3, :], cord[max, :] = cord[max, :].copy(), cord[3, :].copy()
min = np.argmin(np.sum(cord, axis = 1))     
cord[0, :], cord[min, :] = cord[min, :].copy(), cord[0, :].copy()
if cord[1][1] < cord[2][1]:
    cord[1, :], cord[2, :] = cord[2, :].copy(), cord[1, :].copy()

print(cord.tolist())

origin_cord = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)
my_cord = np.array(cord, dtype=np.float32)

# print(cord)
# img_circle = img.copy()
# for i in range(len(cord)):
#     c = cord[i]
#     cv2.circle(img_circle, c, 5, (0, 0, 100 + 50 * i), 3)
# img_circle = cv2.resize(img_circle, (int(img_circle.shape[0] * 0.7), int(img_circle.shape[1] * 0.7)))
# cv2.imshow('circle', img_circle)

M = cv2.getPerspectiveTransform(my_cord, origin_cord)
# M2, mask = cv2.findHomography(my_cord, origin_cord)

persp = cv2.warpPerspective(thres, M, (w, h))

one_or_zero = np.zeros((6, 6))
for i in range(w):
    for j in range(h):
        if persp[i][j] == 255:
            one_or_zero[(i // 8)][(j // 8)] += 1
one_or_zero = one_or_zero / 64

num = 0
for i in range(1, 5):
    for j in range(1, 5):
        num *= 2
        if one_or_zero[i][j] >= 0.5: num += 1
        else: num += 0
print(num)
print(f'{num:016b}')
print('1101010111110000')
cv2.imshow('result', persp)
cv2.waitKey(0)