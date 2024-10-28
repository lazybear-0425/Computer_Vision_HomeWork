import cv2

img = cv2.imread('1021/example/block.jpg')
h = img.shape[1]
w = img.shape[0]
rate = 0.65

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 311, 0)
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('otsu', cv2.resize(otsu, (int(h * rate), int(w * rate))))
cv2.imshow('adaptive', cv2.resize(adaptive, (int(h * rate), int(w * rate))))
cv2.waitKey(0)