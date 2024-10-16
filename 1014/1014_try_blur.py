import cv2
import sys, os

img = cv2.imread('1014/example/people.png')
blur_5_5 = cv2.GaussianBlur(img, (5, 5), 5)
blur_5_55 = cv2.GaussianBlur(img, (5, 5), 55)
blur_55_5 = cv2.GaussianBlur(img, (55, 55), 5)

cv2.imshow('kernel = 5, sigmaX = 5', blur_5_5)
cv2.imshow('kernel = 5, sigmaX = 55', blur_5_55)
cv2.imshow('kernel = 55, sigmaX = 5', blur_55_5)
cv2.waitKey(0)

try:
    sys.path.append(os.getcwd())
    import mytools
    mytools.save_file('1014/result/hk1', [['blur_5_5.png', blur_5_5],
                                          ['blur_5_55.png', blur_5_55],
                                          ['blur_55_5.png', blur_55_5]])
except Exception as e:
    print('存檔失敗QQ，原因:', e)