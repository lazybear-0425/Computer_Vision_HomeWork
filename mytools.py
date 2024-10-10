def save_file(dir, filename):
    import os
    import cv2
    try:
        os.mkdir(dir)
    except: pass
    for file, img in filename:
        file_path = os.path.join(dir, file)
        cv2.imwrite(file_path, img)
        