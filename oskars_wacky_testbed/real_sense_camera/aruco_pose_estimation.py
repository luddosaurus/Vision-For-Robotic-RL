import numpy as np
import cv2

import pyrealsense2 as rs
from aruco_marker_set import MarkerSet

pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Get device product line for setting a supporting resolution
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


# def aruco_display(corners, ids, rejected, image):

# 	if len(corners) > 0:

# 		ids = ids.flatten()

# 		for (markerCorner, markerID) in zip(corners, ids):

# 			corners = markerCorner.reshape((4, 2))
# 			(topLeft, topRight, bottomRight, bottomLeft) = corners

# 			topRight = (int(topRight[0]), int(topRight[1]))
# 			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
# 			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
# 			topLeft = (int(topLeft[0]), int(topLeft[1]))

# 			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
# 			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
# 			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
# 			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

# 			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
# 			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
# 			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

# 			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
# 				0.5, (0, 255, 0), 2)
# 			print("[Inference] ArUco marker ID: {}".format(markerID))

# 	return image


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    marker_size_cm = 3.4
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

    # cv2.calibrateHandEye


    if len(corners) > 0:
        markerSet = MarkerSet(corners, ids, marker_size_cm, matrix_coefficients, distortion_coefficients, frame)

        frame = markerSet.draw_markers_with_axes()
        distances = markerSet.get_camera_distance_to_markers_via_transform()
        # frame = markerSet.calc_distance_from_each_area_version()

        # for i in range(0, len(ids)): marker_size_cm = 3.4 rvec, tvec, markerPoints =
        # cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size_cm, matrix_coefficients, distortion_coefficients)

        #     cv2.aruco.drawDetectedMarkers(frame, corners) 

        # cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 3.4/2) cv2.putText(
        # bgr_color_image, f"{tvec[0][0][2]}", (int(corners[0][0][1][0]),int(corners[0][0][1][1]) ),
        # cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),2,cv2.LINE_AA,)

    return frame


aruco_type = "DICT_4X4_50"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters()

with np.load('./calib_data_2_new/MultiMatrix.npz') as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]
    print(f'intrinsic: {intrinsic_camera}\ndistortion: {distortion}')

for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Start streaming
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)

depth = False

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
        # bgr_color_image = cv2.resize(bgr_color_image, (1280, 720))

        gray_image = cv2.cvtColor(rgb_color_image, cv2.COLOR_RGB2GRAY)

        # -----------------------------DEPTH TEST---------------------------------------

        output = pose_estimation(bgr_color_image, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)

        if depth:
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = output.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:

                resized_color_image = cv2.resize(output, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                # output = cv2.cvtColor(resized_color_image, cv2.COLOR_RGB2BGR)
                images = np.hstack((resized_color_image, depth_colormap))

            else:
                images = np.hstack((output, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
        else:
            cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
