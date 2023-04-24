import cv2

import numpy as np
import pyrealsense2 as rs

import charuco_calib_example as charuco

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CALIB_FILE = "MultiMatrix.npz"

with np.load(f'./{CALIB_FILE}') as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]
    print(f'intrinsic: {intrinsic_camera}\ndistortion: {distortion}')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, 30)

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

rvec = np.random.random((3, 1))
tvec = np.random.random((3, 1))

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     recolored_image = cv2.cvtColor(resized_color_image, cv2.COLOR_RGB2BGR)
        #     images = np.hstack((recolored_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))
        image, rvec, tvec = charuco.estimate_charuco_pose(color_image,
                                                          intrinsic_camera, distortion, rvec, tvec)
        print("r-vec: ", rvec)
        print("t-vec: ", tvec)

        # Show images
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()
