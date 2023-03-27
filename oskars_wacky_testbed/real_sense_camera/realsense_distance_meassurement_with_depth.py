import cv2

import numpy as np
import pyrealsense2 as rs
from aruco_marker_set import MarkerSet

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

arucoParams = cv2.aruco.DetectorParameters()


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    # cv2.calibrateHandEye

    marker_size_cm = 0.034
    distances = dict()
    centers = dict()
    if len(corners) > 0:
        markerSet = MarkerSet(corners, ids, marker_size_cm, matrix_coefficients, distortion_coefficients, frame)
        # frame_1 = markerSet.calc_distance_from_each_area_version()
        frame = markerSet.draw_markers_with_axes()
        distances = markerSet.get_camera_distance_to_markers_via_transform()
        centers = markerSet.get_center_of_every_marker()

        # for i in range(0, len(ids)): marker_size_cm = 3.4 rvec, tvec, markerPoints =
        # cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size_cm, matrix_coefficients, distortion_coefficients)

        #     cv2.aruco.drawDetectedMarkers(frame, corners)

        # cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 3.4/2) cv2.putText(
        # bgr_color_image, f"{tvec[0][0][2]}", (int(corners[0][0][1][0]),int(corners[0][0][1][1]) ),
        # cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),2,cv2.LINE_AA,)

    return frame, distances, centers


with np.load('./calib_data_2_new/MultiMatrix.npz') as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]
print(intrinsic_camera, distortion)

config = rs.config()
pipeline = rs.pipeline()

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

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            print("RESHAPING FOR SOME REASON")
            # resized_depth = cv2.resize(depth_colormap, dsize=(color_colormap_dim[1], color_colormap_dim[0]),
            #                            interpolation=cv2.INTER_AREA)
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            color_image = cv2.cvtColor(resized_color_image, cv2.COLOR_RGB2BGR)

        output, distances, centers = pose_estimation(color_image, cv2.aruco.DICT_4X4_50, intrinsic_camera, distortion)
        if len(centers) != 0:
            for key in distances.keys():
                distance_to_point = depth_frame.get_distance(centers[key][0], centers[key][1])
                cv2.circle(output, centers[key], 20, (255, 100, 100), -1)
                cv2.circle(depth_colormap, centers[key], 20, (255, 100, 100), -1)
                print(f'aruco based {key} - {distances[key]}')
                print(f'depth measurement {key} - {distance_to_point}')
            print('----------------------------------------------------')

        images = np.hstack((output, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()
