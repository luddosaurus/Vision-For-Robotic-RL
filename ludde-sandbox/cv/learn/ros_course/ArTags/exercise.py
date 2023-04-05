#!/usr/bin/env python

import cv2
import numpy as np
from cv2 import aruco

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ARTag(object):
    target_height = 0
    target_width = 0

    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        self.bridge_object = CvBridge()
        self.x = 4

    def camera_callback(self, data):

        try:
            # We select bgr8 because its the OpenCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        image = cv_image
        self.target_height, self.target_width = image.shape[:2]

        image = cv2.resize(image, (int(self.target_width * 0.7), int(self.target_height * 0.7)))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aruco
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # Detect the corners and id's
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Detect Markers
        frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

        # Show the markers detected
        cv2.imshow('markers', frame_markers)
        self.paint_center(image, ids, corners)

        cv2.waitKey(1)

    def paint_earth(self, image, params, params_2):
        paint = cv2.imread(
            '/home/user/catkin_ws_old/src/opencv_for_robotics_images/Unit_5/Course_images/Examples/earth.jpg')
        height, width = paint.shape[:2]
        coordinates = np.array([[0, 0], [width, 0], [0, height], [width, height]])

        hom, status = cv2.findHomography(coordinates, params_2)
        warped_image = cv2.warpPerspective(paint, hom, (int(self.target_width * 0.7), int(self.target_height * 0.7)))

        mask = np.zeros([int(self.target_height * 0.7), int(self.target_width * 0.7), 3], dtype=np.uint8)

        # Mask
        cv2.fillConvexPoly(mask, np.int32([params]), (255, 255, 255), cv2.LINE_AA)
        cv2.imshow('black mask', mask)
        # Sub
        substraction = cv2.subtract(image, mask)
        cv2.imshow('substraction', substraction)
        # Add
        addition = cv2.add(warped_image, substraction)
        cv2.imshow('detection', addition)

    def paint_center(self, image, ids, corners):
        coordinates = []

        if ids is None: return
        for i in range(len(ids)):
            c = corners[i][0]
            cv2.circle(image, (int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255, 255, 0), -1)
            coordinates.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

        coordinates = np.array(coordinates)

        # if(len(coordinates)>=4):
        #     coordinates = self.order_coordinates(coordinates)

        if (len(coordinates) >= 4):
            # Sort model_old 1
            coordinates = self.order_coordinates2(coordinates, False)

            # Sort Model 2
            coordinates_2 = self.order_coordinates2(coordinates, True)

            self.paint_earth(image, coordinates, coordinates_2)

        cv2.drawContours(image, [coordinates], -1, (255, 0, 150), -1)
        cv2.imshow('no_conversion', image)

    def order_coordinates(self, pts):
        coordinates = np.zeros((4, 2), dtype="int")

        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]
        coordinates[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]
        coordinates[3] = pts[np.argmax(diff)]

        return coordinates

    def order_coordinates2(self, pts, var):
        coordinates = np.zeros((4, 2), dtype="int")

        if (var):
            # Parameters sort model_old 1
            s = pts.sum(axis=1)
            coordinates[0] = pts[np.argmin(s)]
            coordinates[3] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            coordinates[1] = pts[np.argmin(diff)]
            coordinates[2] = pts[np.argmax(diff)]

        else:
            # Parameters sort model_old 2
            s = pts.sum(axis=1)
            coordinates[0] = pts[np.argmin(s)]
            coordinates[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            coordinates[1] = pts[np.argmin(diff)]
            coordinates[3] = pts[np.argmax(diff)]

        return coordinates

    def area_substraction(self):
        print("hej")


def main():
    load_feature_object = ARTag()
    rospy.init_node('camera_node', anonymous=True)

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
