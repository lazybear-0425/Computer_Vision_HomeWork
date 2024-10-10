import numpy as np
import cv2

def main():
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    src = cv2.imread('1007/example/T_pic.png')
    if src is None:
        print('Error opening image')
        return -1

    src = cv2.GaussianBlur(src, (5, 5), 0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # 去強化邊緣，不加的話之後HoughLinesP會產生很多條線
    _, binary = cv2.threshold(grad, 25, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=70, minLineLength=20, maxLineGap=30)

    if lines is not None:
        print(f'Number of lines detected: {len(lines)}')
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Sobel', grad)
    cv2.imshow('Result', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 存檔
    try:
        import os
        import sys
        sys.path.append(os.getcwd())
        import mytools # from mytools.py
        mytools.save_file('1007/result/hk2', [['T-sobel.png', grad], ['T-sobel-result.png', src]])
    except:
        print('存檔失敗QQ')

if __name__ == "__main__":
    main()
