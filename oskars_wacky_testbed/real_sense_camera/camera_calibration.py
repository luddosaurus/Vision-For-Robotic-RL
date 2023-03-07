import glob
import os

import cv2
import numpy as np
from extract_realsense_parameters import ExtractParameters


#lth board
chessboardSize = (5,8)
size_of_chessboard_squares_mm = 20

# chessboardSize = (6,8)
# size_of_chessboard_squares_mm = 15
calib_data_path = "/home/oskarlarsson/Documents/Programs/Python_tests/real_sense_camera/calib_data_2_new_2"
image_path = '/home/oskarlarsson/Documents/Programs/Python_tests/real_sense_camera/images_real_sense/*.png'



CHECK_DIR = os.path.isdir(calib_data_path)


if not CHECK_DIR:
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')

else:
    print(f'"{calib_data_path}" Directory already Exists.')

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


calibration_images = glob.glob(image_path)

num_used_images = 0

for calibration_image in calibration_images:
    color_image = cv2.imread(calibration_image)
    # print(color_image.shape)
    # cv2.imshow('test', color_image)
    # cv2.waitKey(0)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray_image, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray_image, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(color_image, chessboardSize, corners2, ret)
        cv2.imshow('img', color_image)
        key = cv2.waitKey(0)

        if key == ord("y"):
            objpoints.append(objp)
            
            imgpoints.append(corners)
            num_used_images += 1
        # Draw and display the corners

        # cv2.waitKey(1000)
print(num_used_images)

cv2.destroyAllWindows()

############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
# pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
# pickle.dump(dist, open( "dist.pkl", "wb" ))
print(f'camera_matrix: {cameraMatrix}\ndist: {dist}')
np.savez(
    f"{calib_data_path}/MultiMatrix",
    camMatrix=cameraMatrix,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)


############## UNDISTORTION #####################################################

img = cv2.imread('./images_real_sense/image5.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# Undistort
dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('caliResult1.png', dst)



# Undistort with Remapping
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('caliResult2.png', dst)




# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
extract_params = ExtractParameters()
extract_params.print_parameters()