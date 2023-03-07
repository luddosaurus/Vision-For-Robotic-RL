import numpy as np
import cv2
from cv2 import aruco as aruco
from image_transform import *

import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

# ArUco Marker Settings
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
param_markers = aruco.DetectorParameters()

camera_width = 640
camera_height = 480


def find_markers(img, debug=False):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get corners (in pixel coordinates) id and rejected
    marker_corners, marker_ids, rejected = aruco.detectMarkers(
        image=gray_image,
        dictionary=marker_dict,
        parameters=param_markers
    )
    image_markers = aruco.drawDetectedMarkers(image=img, corners=marker_corners, ids=marker_ids)

    if debug:
        print(marker_corners)

    return image_markers, marker_corners, marker_ids


def get_first_corners(corner_array):
    return corner_array[0]


# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
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
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        recolored_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)


        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        image, corners, ids = find_markers(recolored_image, debug=False)
        warp_image = image
        if ids is not None:
            # for i in range(len(ids)):
            first_corners = corners[0]
            warp_image = get_warp(image, first_corners, camera_height, camera_width)
                # cv2.imshow("Warp" + str(i), cv2.resize(warp_image, (0,0), fx=0.3, fy=0.3))
        
        images = np.hstack((recolored_image, warp_image))
        cv2.imshow("stuff", images)

        # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     recolored_image = cv2.cvtColor(resized_color_image, cv2.COLOR_RGB2BGR)
            
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()

# internal_camera = 0
# external_camera = 1

# cap = cv2.VideoCapture(internal_camera)

# # Camera Size
# camera_width = 640
# camera_height = 480
# cap.set(3, camera_width)
# cap.set(4, camera_height)
# cap.set(10, 100)

# # ArUco Marker Settings
# marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# param_markers = aruco.DetectorParameters()


# def find_markers(img, debug=False):
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Get corners (in pixel coordinates) id and rejected
#     marker_corners, marker_ids, rejected = aruco.detectMarkers(
#         image=gray_image,
#         dictionary=marker_dict,
#         parameters=param_markers
#     )
#     image_markers = aruco.drawDetectedMarkers(image=img, corners=marker_corners, ids=marker_ids)

#     if debug:
#         print(marker_corners)

#     return image_markers, marker_corners, marker_ids


# def get_first_corners(corner_array):
#     return corners[0]


# while True:
#     # Run Camera
#     success, image = cap.read()
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     image, corners, ids = find_markers(image, debug=False)
#     if ids is not None:
#         first_corners = get_first_corners(corners)
#         warp_image = get_warp(image, first_corners, camera_height, camera_width)
#         cv2.imshow("Warp", warp_image)

#     cv2.imshow("Video", image)


# cap.release()
# cv2.destroyAllWindows()