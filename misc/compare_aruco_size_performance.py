import numpy as np
import cv2

import pyrealsense2 as rs
from aruco_marker_set import MarkerSet

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

with np.load('./MultiMatrix.npz') as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]
    print(f'intrinsic: {intrinsic_camera}\ndistortion: {distortion}')

for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

frames_counted = 0
frames_detected_small = 0
frames_detected_large = 0
frames_detected_5x5 = 0


def pose_estimation(frame, aruco_dict_type, marker_size, matrix_coefficients, distortion_coefficients):
    global frames_detected_small
    global frames_detected_large
    global frames_detected_5x5
    global frames_counted

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

    if len(corners) > 0:
        markerSet = MarkerSet(corners, ids, marker_size, matrix_coefficients, distortion_coefficients, frame)

        frame = markerSet.draw_markers_with_axes()
        distances = markerSet.get_camera_distance_to_markers_via_transform()
        # print(distances)
        if marker_size == 0.034 and 2 in ids:
            # print("in small ", marker_size)
            frames_detected_small += 1

        if aruco_dict_type == ARUCO_DICT["DICT_4X4_50"] and 0 in ids:
            # print("in large ", marker_size)
            frames_detected_large += 1

        if aruco_dict_type == ARUCO_DICT["DICT_5X5_50"] and 0 in ids:
            frames_detected_5x5 += 1

    return frame, ids


pipeline.start(config)
try:

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays

        rgb_color_image = np.asanyarray(color_frame.get_data())

        bgr_color_image = cv2.cvtColor(rgb_color_image, cv2.COLOR_RGB2BGR)
        # bgr_color_image = cv2.resize(bgr_color_image, (1280, 720))

        gray_image = cv2.cvtColor(rgb_color_image, cv2.COLOR_RGB2GRAY)

        # -----------------------------DEPTH TEST---------------------------------------

        # --------4x4 small--------
        aruco_type = "DICT_4X4_50"
        arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
        arucoParams = cv2.aruco.DetectorParameters()

        output, ids = pose_estimation(rgb_color_image, ARUCO_DICT[aruco_type], 0.034, intrinsic_camera,
                                      distortion)
        output, ids = pose_estimation(output, ARUCO_DICT[aruco_type], 0.1, intrinsic_camera,
                                      distortion)

        aruco_type = "DICT_5X5_50"
        arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
        arucoParams = cv2.aruco.DetectorParameters()
        output, ids = pose_estimation(output, ARUCO_DICT[aruco_type], 0.1, intrinsic_camera,
                                      distortion)

        frames_counted += 1
        print(f'small: {frames_detected_small}/{frames_counted}')
        print(f'large: {frames_detected_large}/{frames_counted}')
        print(f'5x5: {frames_detected_5x5}/{frames_counted}')
        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
