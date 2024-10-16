import cv2
import numpy as np
import os
import sys
import tkinter as tk 
# 教學 : https://steam.oxxostudio.tw/category/python/tkinter/index.html

last_x = -1
last_y = -1
kernel_size = 5
sigmx = 5
close_window = False

root = tk.Tk()
root.geometry('300x200+950+365')
tk_kernel_size = tk.StringVar(); tk_kernel_size.set(str(kernel_size))
tk_sigmx = tk.StringVar(); tk_sigmx.set(str(sigmx))

def mouse_event(event, x, y, flags, userdata):
    global last_x
    global last_y
    global kernel_size
    global sigmx
    
    if event == cv2.EVENT_LBUTTONDOWN:    
        last_x = x
        last_y = y
    elif event == cv2.EVENT_MOUSEMOVE and last_x != -1 and last_y != -1:
        userdata = np.copy(userdata)
        cv2.rectangle(userdata, (last_x, last_y), (x, y), (0, 0, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        # cv2.rectangle(userdata, (last_x, last_y), (x, y), (0, 0, 255), 2)
        if last_x > x: x,last_x = last_x, x
        if last_y > y: y,last_y = last_y, y
        roi = np.copy(userdata[last_y:y, last_x:x, :])
        roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), sigmx)
        userdata[last_y:y, last_x:x, :] = roi
        # init
        last_x = -1
        last_y = -1
    cv2.imshow('result', userdata)
        
def modify():
    global kernel_size
    global sigmx
    global tk_kernel_size
    global tk_sigmx
    try:
        kernel_size = int(tk_kernel_size.get())
        sigmx = int(tk_sigmx.get())
    except Exception as e:
        print('請輸入數字!')
        # print(e) # for debug
    else:
        print('修改成功!')
        
def close():
    global close_window
    root.destroy()
    close_window = True

img = cv2.imread('1014/example/people.png')
copy_img = img.copy()

cv2.imshow('result', img)
cv2.setMouseCallback('result', mouse_event, img)
cv2.moveWindow('result', 550, 300)

label_kernel_size = tk.Label(root, text='Enter kernel size:', font='3px').pack()
entry_kernel_size = tk.Entry(root, textvariable=tk_kernel_size).pack()
label_sigmx = tk.Label(root, text='Enter sigmaX', font='3px').pack()
entry_sigmx = tk.Entry(root, textvariable=tk_sigmx).pack()
button_modify = tk.Button(root, text='OK', command=modify, width=6).pack()
button_close = tk.Button(root, text='close', command=close, width=6).pack()
root.mainloop()

while True:
    cv2.imshow('result', img)
    key = cv2.waitKey(30)
    if key == ord('q') or key == 27 or close_window: 
        cv2.destroyAllWindows()
        break

try:
    sys.path.append(os.getcwd())
    import mytools
    mytools.save_file('1014/result/hk1', [['blur+roi_advanced.png', img]])
except Exception as e:
    print('存檔失敗QQ，原因:', e)