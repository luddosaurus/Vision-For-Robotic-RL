import numpy as np
import cv2
from cv2 import aruco as aruco
import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()

chessboardSize = (7,10)
size_of_chessboard_squares_mm = 15
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False


with np.load('./calib_data_2/MultiMatrix_720.npz') as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix','distCoef','rVector','tVector')]


for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Start streaming
pipeline.start(config)

depth = True

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        rgb_color_image = np.asanyarray(color_frame.get_data())
        bgr_color_image = cv2.cvtColor(rgb_color_image, cv2.COLOR_RGB2BGR)
        
        gray_image = cv2.cvtColor(rgb_color_image, cv2.COLOR_RGB2GRAY)


        #-----------------------------DEPTH TEST---------------------------------------





        ret, corners = cv2.findChessboardCorners(gray_image, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray_image, corners, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(bgr_color_image, chessboardSize, corners2, ret)



        if depth:
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = bgr_color_image.shape


            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                
                resized_color_image = cv2.resize(bgr_color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                # output = cv2.cvtColor(resized_color_image, cv2.COLOR_RGB2BGR)
                images = np.hstack((resized_color_image, depth_colormap))
                
            else:
                images = np.hstack((bgr_color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
        else: 
            cv2.imshow('Estimated Pose', bgr_color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()