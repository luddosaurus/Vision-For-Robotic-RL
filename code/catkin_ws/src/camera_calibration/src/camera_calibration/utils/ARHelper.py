# from cv2 import aruco as aruco
import cv2
import numpy as np


class ARHelper:
    # ArUco Marker Settings
    # marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # param_markers = cv2.aruco.DetectorParameters()

    marker_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    param_markers = cv2.aruco.DetectorParameters_create()

    def __init__(self, marker_size=50):
        self.marker_size = marker_size

    def find_markers(self, img, debug=False):


        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get corners (in pixel coordinates) id and rejected
        marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
            image=gray_image,
            dictionary=self.marker_dict,
            parameters=self.param_markers,
        )



        marked_image = cv2.aruco.drawDetectedMarkers(
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

    @staticmethod
    def draw_vectors(img, marker_corners, marker_ids, matrix, distortion, marker_size=0.034):

        axis_length = 0.02
        if len(marker_corners) > 0:
            for i in range(0, len(marker_ids)):
                rotation_vec, translation_vec, marker_points = cv2.aruco.estimatePoseSingleMarkers(
                    corners=marker_corners[i],
                    markerLength=marker_size,
                    cameraMatrix=matrix,
                    distCoeffs=distortion
                )

                if rotation_vec is not None:
                    # print("Rotation: ", rotation_vec)
                    # z - blue , y - green, x - red
                    cv2.drawFrameAxes(
                        image=img,
                        cameraMatrix=matrix,
                        distCoeffs=distortion,
                        rvec=rotation_vec,
                        tvec=translation_vec,
                        length=axis_length)

        return img

    @staticmethod
    def draw_vectors(img, intrinsic_camera_matrix, distortion, rotation_vecs, translation_vecs):
        axis_length = 0.02
        for (r_vec, t_vec) in zip(rotation_vecs, translation_vecs):
            # z - blue , y - green, x - red
            cv2.drawFrameAxes(
                image=img,
                cameraMatrix=intrinsic_camera_matrix,
                distCoeffs=distortion,
                rvec=r_vec,
                tvec=t_vec,
                length=axis_length)

    @staticmethod
    def estimate_charuco_pose(image, camera_matrix, dist_coefficients, rvec, tvec):
        square_length = 0.015  # mm
        marker_length = 0.012  # mm
        squares_x = 9
        squares_y = 7

        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_length,
            marker_length, cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is None:
            return image, rvec, tvec
        cv2.aruco.drawDetectedMarkers(image, corners)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        # charuco_findings = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        # print(charuco_findings)
        if charuco_corners is None or charuco_ids is None:
            return image, rvec, tvec

        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners=charuco_corners, charucoIds=charuco_ids,
                                                                board=board, cameraMatrix=camera_matrix,
                                                                distCoeffs=dist_coefficients, rvec=rvec, tvec=tvec,
                                                                useExtrinsicGuess=True)

        if rvec is not None and tvec is not None:
            cv2.drawFrameAxes(image=image, cameraMatrix=camera_matrix, distCoeffs=dist_coefficients, rvec=rvec,
                              tvec=tvec,
                              length=0.03, thickness=2)

        return image, rvec, tvec
