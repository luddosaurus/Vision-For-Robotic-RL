import numpy as np
import cv2
from cv2 import aruco as aruco
from base.utils.ARHelper import *


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


internal_camera = 0
external_camera = 1

cap = cv2.VideoCapture(external_camera)

# Camera Size
camera_width = 640
camera_height = 480
cap.set(3, camera_width)
cap.set(4, camera_height)
cap.set(10, 100)

# ArUco Marker Settings

parameters = aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

detector = ObjectDetector()


def draw_objects(img, ratio):
    contours = detector.detect_objects(image)
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


while True:
    # Run Camera
    success, image = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Use marker to find scale
    corners, _, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    int_corners = np.int0(corners)
    cv2.polylines(image, int_corners, True, (0, 255, 0), 5)

    aruco_perimeter = cv2.arcLength(corners[0], True)
    pixel_cm_ratio = aruco_perimeter / 20

    image = draw_objects(image, pixel_cm_ratio)

    cv2.imshow("Video", image)


cap.release()
cv2.destroyAllWindows()

