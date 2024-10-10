import sys
import cv2
import numpy as np

def mouse_event(event, x, y, flags, param):
    img, origin = param # img -> 有畫圈的圖, origin -> 原圖
    w = 70
    h = 50
    # 調整亮度=====================================================
    # 參考 : https://steam.oxxostudio.tw/category/python/ai/opencv-adjust.html
    contrast = 0
    brightness = 100
    # 避免 uint8 造成 overflow
    process = np.int16(img[y - h : y + h, x-w : x + w, :]) * (contrast / 127 + 1) - contrast + brightness
    process = np.clip(process, 0, 255) # 限縮範圍
    process = np.uint8(process)
    origin[y - h : y + h, x-w : x + w, :] = process
    # 調整亮度=====================================================
    cv2.rectangle(origin, (x - w, y - h), (x + w, y + h), (0, 255, 255), 1)
    cv2.imshow('circle', origin)
    
def show_pic(img):
    origin_img = np.copy(img) # 拷貝一份原圖之後用
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
        cv2.circle(img, center, radius, (155, 0, 155), 3)

    num_circle = circles.shape[1]
    text = f'number of circles: {num_circle}'
    cv2.putText(origin_img, text, (img.shape[0] - 60, img.shape[1] - 145), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 100, 0), 1)
    cv2.putText(img, text, (img.shape[0] - 60, img.shape[1] - 145), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (225, 70, 0), 1)
    cv2.imshow('circle', origin_img)
    cv2.setMouseCallback('circle', mouse_event, (img, origin_img)) # 這句要等window建好才能用QQ
    
    # 存檔
    try:
        import os
        sys.path.append(os.getcwd())
        import mytools # from mytools.py
        mytools.save_file('1007/result/hk3', [['coin-circle-advanced.png', origin_img]])
    except:
        print('存檔失敗QQ')
    
if __name__ == '__main__':
    img = cv2.imread('1007/example/coin.jpg', cv2.IMREAD_COLOR)
    while True:
        show_pic(np.copy(img))
        key = cv2.waitKey(30)
        '''按下q或esc關閉'''
        if key == 27 or key == ord('q'): break
