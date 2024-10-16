def save_file(dir, filename):
    import os
    import cv2
    try:
        # 使用mkdir若沒有父目錄會出問題
        os.makedirs(dir, exist_ok=True) 
    except: pass
    for file, img in filename:
        file_path = os.path.join(dir, file)
        cv2.imwrite(file_path, img)
        