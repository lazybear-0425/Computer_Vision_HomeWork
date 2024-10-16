import cv2
import numpy as np
import os
 
img = cv2.imread('1007/example/word.png', cv2.IMREAD_GRAYSCALE)

threshold = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,2)
cv2.imshow('threshold', threshold)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3))
''' 因為是白底黑字，所以erode和dilate要相反操作! '''
erode = cv2.erode(threshold, kernel)
dilate = cv2.dilate(erode, kernel)
cv2.imshow('open', dilate)
''' 用膨脹去變成連線 '''
kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
erode_line = cv2.erode(dilate, kernel_dil)
cv2.imshow('dilate', erode_line)

cv2.waitKey(0) 

# 存檔
try:
    import os
    import sys
    sys.path.append(os.getcwd())
    import mytools # from mytools.py
    mytools.save_file('1007/result/hk1', 
                      [['word-binary.png', threshold], 
                       ['word-open.png', dilate], 
                       ['word-dilate.png', erode_line]])
except:
    print('存檔失敗QQ')
'''
第一張 二質化
去雜訊
連起來
'''