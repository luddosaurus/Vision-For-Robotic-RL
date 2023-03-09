import cv2
from cv2 import aruco
import numpy as np
import pyrealsense2 as rs
from aruco_marker_set import MarkerSet

config = rs.config()
pipeline = rs.pipeline()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

arucoDict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

arucoParams = cv2.aruco.DetectorParameters()


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)
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

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            # resized_depth = cv2.resize(depth_colormap, dsize=(color_colormap_dim[1], color_colormap_dim[0]),
            #                            interpolation=cv2.INTER_AREA)
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            color_image = cv2.cvtColor(resized_color_image, cv2.COLOR_RGB2BGR)

        output, distances, centers = pose_estimation(color_image, aruco.DICT_4X4_50, intrinsic_camera, distortion)
        if len(centers) != 0:
            distance_to_4 = depth_frame.get_distance(centers[4][0], centers[4][1])
            print(f'depth measurement - {distance_to_4}')
            print(f'aruco based - {distances[4]}')

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
