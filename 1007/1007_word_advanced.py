import cv2 as cv2
import numpy as np
import os
 
img = cv2.imread('1007/example/word.png', cv2.IMREAD_GRAYSCALE)
# 可以手動調整
adjust_window = 'adjust'
blocksize = 3
C = 2
adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

def adjust_blocksize(val):
    global blocksize
    blocksize = val
    # 這邊要注意 blocksize % 2 == 1 and blocksize > 1
    if(blocksize % 2 == 0): blocksize += 1
    if(blocksize <= 1): blocksize = 3
    cv2.setTrackbarPos('blocksize', adjust_window, blocksize)
    
def adjust_C(val):
    global C
    C = val
    cv2.setTrackbarPos('C', adjust_window, C)

def adjust_method(val):
    global adaptiveMethod
    if(val == 0) : adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
    else : adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    cv2.setTrackbarPos('method', adjust_window, val)

cv2.namedWindow(adjust_window) # 新增一個空白視窗
#       parameter : 
cv2.createTrackbar('C', adjust_window, C, 30, adjust_C)
cv2.createTrackbar('blocksize', adjust_window, blocksize, 255, adjust_blocksize)
cv2.createTrackbar('method', adjust_window, 1, 1, adjust_method)

while True:
    threshold = cv2.adaptiveThreshold(img, 255, adaptiveMethod,
            cv2.THRESH_BINARY,blocksize,C) # blocksize, C

    cv2.imshow(adjust_window, threshold)
    cv2.imshow('threshold', threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 3))
    ''' 因為是白底黑字，所以erode和dilate要相反操作! '''
    erode = cv2.erode(threshold, kernel)
    dilate = cv2.dilate(erode, kernel)
    cv2.imshow('open', dilate)
    ''' 用膨脹去變成連線 '''
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 4))
    erode_line = cv2.erode(dilate, kernel_dil)
    cv2.imshow('dilate', erode_line)

    key = cv2.waitKey(30) 
    if key == ord('q') or key == 27: # esc == 27
        break

# 存檔
# try:
#     import os
#     import sys
#     sys.path.append(os.getcwd())
#     import mytools # from mytools.py
#     mytools.save_file('1007/result/hk1', 
#                       [['word-binary.png', threshold], 
#                        ['word-open.png', dilate], 
#                        ['word-dilate.png', erode_line]])
# except:
#     print('存檔失敗QQ')
'''
第一張 二質化
去雜訊
連起來
'''