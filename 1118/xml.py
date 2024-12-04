#%%
import cv2
import numpy as np

# ===================擷取xml=======================================
# 參考自 : https://docs.opencv.org/4.x/dd/d74/tutorial_file_input_output_with_xml_yml.html 

file = cv2.FileStorage('1118/out_camera_data.xml', cv2.FileStorage_READ)

# imagePoints = file.getNode('image_points').mat()
camera_matrix = file.getNode('camera_matrix').mat()
distCoeffs = file.getNode('distortion_coefficients').mat()

gridPoints = file.getNode('grid_points')
pcols = file.getNode('board_width').real()
prows = file.getNode('board_height').real()
ssize = file.getNode('square_size').real()

file.release()

objectPoints = np.zeros((9 * 6, 3), np.float64)
for r in range(prows):
    for c in range(pcols):
        i = r * pcols + c
        objectPoints[i] = [gridPoints.at(i * 3).real(), gridPoints[i * 3 + 1], gridPoints[i * 3 + 2]]

#%%
# ===================使用slovePnP=======================================

img = cv2.imread('1118/pic1.jpg')
# 參數說明
# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
retval, imagePoints = cv2.findChessboardCorners(img, (9, 6))

objectPoints = np.zeros((9 * 6, 3), np.float32)
objectPoints[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
objectPoints *= 50  # 每個方格的實際大小(米)

# ===================使用slovePnP=======================================
# cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs) ->	retval, rvec, tvec
# print(imagePoints.shape)

retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, camera_matrix, distCoeffs)
# print(retval)
# print(rvec)
# print(tvec)

dst, jacobian = cv2.Rodrigues(rvec)
# print(dst.shape) # shape : (3, 3)
# print(jacobian.shape) # shape : (3, 9)