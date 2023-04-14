import numpy as np
import cv2
from cv2 import aruco as aruco
import os
import pyrealsense2 as rs

# Setup
pipeline = rs.pipeline()
config = rs.config()

CHESS_BOARD_DIM = (7, 10)
image_counter = 0

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
found_rgb = False

# checking if  images dir is exist not, if not then create images directory
image_dir_path = "./images_real_sense_1920"

CHECK_DIR = os.path.isdir(image_dir_path)
# if directory does not exist create
if not CHECK_DIR:
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created')
else:
    print(f'"{image_dir_path}" Directory already Exists.')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ArUco Marker Settings
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
param_markers = aruco.DetectorParameters()


def detect_checker_board(image, grayImage, criteria, boardDimension):
    ret, corners = cv2.findChessboardCorners(grayImage, boardDimension)

    if ret == True:
        corners1 = cv2.cornerSubPix(grayImage, corners, (3, 3), (-1, -1), criteria)
        image = cv2.drawChessboardCorners(image, boardDimension, corners1, ret)

    return image, ret


for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        rgb_color_image = np.asanyarray(color_frame.get_data())
        bgr_color_image = cv2.cvtColor(rgb_color_image, cv2.COLOR_RGB2BGR)
        bgr_color_image_copy = bgr_color_image.copy()
        gray_image = cv2.cvtColor(rgb_color_image, cv2.COLOR_RGB2GRAY)

        image, board_detected = detect_checker_board(bgr_color_image, gray_image, criteria, CHESS_BOARD_DIM)
        cv2.putText(
            bgr_color_image,
            f"saved_img : {image_counter}",
            (30, 40),
            cv2.FONT_HERSHEY_PLAIN,
            1.4,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        # Get the image dimensions
        height, width = bgr_color_image.shape[:2]

        # Downscale the image by 50%
        new_width = int(width / 2)
        new_height = int(height / 2)

        cv2.imshow("Capture window", cv2.resize(bgr_color_image, (new_width, new_height)))

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # depth_colormap_dim = depth_colormap.shape
        # color_colormap_dim = rgb_color_image.shape

        # images = np.hstack((recolored_image, depth_image))
        # cv2.imshow("stuff", images)

        # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     recolored_image = cv2.cvtColor(resized_color_image, cv2.COLOR_RGB2BGR)
        #     # images = np.hstack((recolored_image, depth_colormap))

        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', bgr_color_image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("s") and board_detected == True:
            # storing the checker board image
            cv2.imwrite(f"{image_dir_path}/image{image_counter}.png", bgr_color_image_copy)
            print(f"saved image number {image_counter}")
            image_counter += 1  # incrementing the image counter

finally:

    # Stop streaming
    pipeline.stop()
