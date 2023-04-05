#!/usr/bin/env python

import numpy as np
import cv2
from cv2 import aruco


class ARDetect():
    image_height = 0
    image_width = 0
    paint_everything = False
    wanted_image = '/home/user/catkin_ws_old/src/opencv_for_robotics_images/Project/Course_images/wanted.png'

    def find_tag(self, image):
        self.image_height, self.image_width = image.shape[:2]

        resize_factor = 0.7
        new_size = (int(self.image_width * resize_factor), int(self.image_height * resize_factor))
        image = cv2.resize(image, new_size)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids = self.find_parameters(gray_image)
        self.paint_center(image, ids, corners)

    def find_parameters(self, image):

        # Aruco
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # Detect the corners and id's
        corners, ids, rejectedImgPoints = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

        if self.paint_everything:
            # Detect Markers
            frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

            # Show the markers detected
            cv2.imshow('markers', frame_markers)

        return corners, ids

    def paint_wanted(self, image, params, params_2):
        paint = cv2.imread(self.wanted_image)
        height, width = paint.shape[:2]
        coordinates = np.array([[0, 0], [width, 0], [0, height], [width, height]])

        hom, status = cv2.findHomography(coordinates, params_2)
        warped_image = cv2.warpPerspective(paint, hom, (int(self.image_width * 0.7), int(self.image_height * 0.7)))

        # Important order here
        mask = np.zeros([int(self.image_height * 0.7), int(self.image_width * 0.7), 3], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32([params]), (255, 255, 255), cv2.LINE_AA)
        substraction = cv2.subtract(image, mask)
        addition = cv2.add(warped_image, substraction)

        if self.paint_everything:
            cv2.imshow('black mask', mask)
            cv2.imshow('substraction', substraction)

        cv2.imshow('detection', addition)

    def paint_center(self, image, ids, corners):
        coordinates = []

        if ids is None: return
        for i in range(len(ids)):
            c = corners[i][0]
            cv2.circle(image, (int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255, 255, 0), -1)
            coordinates.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

        coordinates = np.array(coordinates)

        if (len(coordinates) >= 4):
            coordinates = self.order_coordinates(coordinates, False)
            coordinates_2 = self.order_coordinates(coordinates, True)
            self.paint_wanted(image, coordinates, coordinates_2)

        cv2.drawContours(image, [coordinates], -1, (255, 0, 150), -1)
        cv2.imshow('no_conversion', image)

    def order_coordinates(self, pts, var):
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
