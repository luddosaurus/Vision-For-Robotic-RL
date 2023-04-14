from cv2 import aruco as aruco
import cv2
import numpy as np


class ARHelper:

    # ArUco Marker Settings
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    param_markers = aruco.DetectorParameters()

    def __init__(self, marker_size=50):
        self.marker_size = marker_size

    def find_markers(self, img, debug=False):

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get corners (in pixel coordinates) id and rejected
        marker_corners, marker_ids, rejected = aruco.detectMarkers(
            image=gray_image,
            dictionary=self.marker_dict,
            parameters=self.param_markers,
        )

        marked_image = aruco.drawDetectedMarkers(
            image=img,
            corners=marker_corners,
            ids=marker_ids)

        if debug:
            print(marker_corners)

        return marked_image, marker_corners, marker_ids

    @staticmethod
    def find_center(corners, marker_ids):

        # centers = []
        # for index in range(0, len(marker_ids)):
        points_arr = np.array(corners)
        center = np.mean(points_arr, axis=(0, 1)).astype(int)
        # centers.append(center)

        return center

    @staticmethod
    def get_camera_pose(camera_matrix, distortion_coefficients, image_points, world_points):

        image_points = np.array(image_points, dtype=np.float64)
        world_points = np.array(world_points, dtype=np.float64)


        # undistort image points
        # image_points = cv2.undistortPoints(image_points,
        #                                   camera_matrix,
        #                                   distortion_coefficients)

        # estimate camera pose
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(world_points,
                                          image_points,
                                          camera_matrix,
                                          distortion_coefficients)

        return tvec, rvec

