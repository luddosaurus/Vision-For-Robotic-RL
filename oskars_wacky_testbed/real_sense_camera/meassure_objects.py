import cv2
from cv2 import aruco
import numpy as np
import pyrealsense2 as rs
# from ARHelper import *

class ObjectDetector:
    def __init__(self):
        pass

    @staticmethod
    def detect_objects(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                objects_contours.append(cnt)

        return objects_contours
    
parameters = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def draw_objects(img, ratio):
    contours = detector.detect_objects(img)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect

        # Get Width and Height of the Objects by applying the Ratio pixel to cm
        object_width = w / ratio
        object_height = h / ratio

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.polylines(img, [box], True, (255, 0, 0), 2)

        cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

    return img


detector = ObjectDetector()

pipeline = rs.pipeline()
config = rs.config()

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

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays

        color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)

        # Use marker to find scale
        corners, ids, _ = aruco.detectMarkers(color_image, aruco_dict, parameters=parameters)
        if len(corners) > 0 :
            int_corners = np.intp(corners)
            cv2.polylines(color_image, int_corners, True, (0, 255, 0), 5)

            aruco_perimeter = cv2.arcLength(corners[0], True)
            pixel_cm_ratio = aruco_perimeter / 15

            color_image = draw_objects(color_image, pixel_cm_ratio)

        cv2.imshow("Video", color_image)

        # # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()