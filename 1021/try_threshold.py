import cv2

img = cv2.imread('1021/example/block.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)
_, thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate = cv2.erode(thres, kernel)

cv2.imwrite('1021/result/threshold_1.jpg', dilate)