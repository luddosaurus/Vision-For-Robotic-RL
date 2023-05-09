#! /usr/bin/python3.8
from itertools import combinations

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg

import cv2
import os
import numpy as np

from camera_calibration.utils.ARHelper import ARHelper
from camera_calibration.utils.extract_realsense_parameters import ExtractParameters

CALIB_DATA_NAME = "calib_data_1280"


class InternalCalibrator(object):
    def __init__(self, board_dimensions, square_length, marker_length):
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.my_callback)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard_create(board_dimensions[0], board_dimensions[1], square_length,
                                                   marker_length, self.aruco_dict)
        self.board_dimensions = board_dimensions
        self.size_of_chessboard_squares = square_length
        self.marker_length = marker_length

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.saved_images = list()
        self.calib_data_path = os.path.join(rospkg.RosPack().get_path('camera_calibration'),
                                            'calibration_data/internal_calibration')

        self.num_images = 20

    def my_callback(self, image_message):

        try:
            image = self.bridge.imgmsg_to_cv2(img_msg=image_message, desired_encoding='bgr8')
        except CvBridgeError:
            print(CvBridgeError)
        original_image = image.copy()
        # _, board_detected = self.detect_checkerboard(image=image)
        board_detected, image = ARHelper.detect_and_draw_charuco(image, square_length=self.size_of_chessboard_squares,
                                                                 marker_length=self.marker_length,
                                                                 rows=self.board_dimensions[0],
                                                                 columns=self.board_dimensions[1])

        if len(self.saved_images) < self.num_images:
            cv2.imshow('image subscriber', image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            rospy.signal_shutdown('We are done here')
        elif key == ord('s') and board_detected:
            self.saved_images.append(original_image)

        calibration_finished = False
        if len(self.saved_images) >= self.num_images:
            # calibration_finished = self.calibrate()
            calibration_finished = self.calibrate_charuco()
        if calibration_finished:
            # extract_params = ExtractParameters()
            # extract_params.print_parameters()
            rospy.signal_shutdown('We are done here')

    def detect_checkerboard(self, image):

        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_scale_image, self.board_dimensions, None)

        if ret:
            corners1 = cv2.cornerSubPix(gray_scale_image, corners, (3, 3), (-1, -1), self.criteria)
            image = cv2.drawChessboardCorners(image, self.board_dimensions, corners1, ret)

        return image, ret

    def calibrate(self):
        objp = np.zeros((self.board_dimensions[0] * self.board_dimensions[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_dimensions[0], 0:self.board_dimensions[1]].T.reshape(-1, 2)

        objp = objp * self.size_of_chessboard_squares_mm

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        num_used_images = 0

        for calibration_image in self.saved_images:
            # color_image = cv2.imread(calibration_image)
            # print(color_image.shape)
            # cv2.imshow('test', calibration_image)
            # cv2.waitKey(0)
            gray_image = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray_image, self.board_dimensions, None)

            # If found, add object points, image points (after refining them)
            if ret:
                corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), self.criteria)
                cv2.drawChessboardCorners(calibration_image, self.board_dimensions, corners2, ret)
                # cv2.imshow('img', calibration_image)
                # key = cv2.waitKey(0)

                # if key == ord("y"):
                objpoints.append(objp)

                imgpoints.append(corners)
                num_used_images += 1
                # Draw and display the corners

                # cv2.waitKey(1000)
        print(num_used_images)

        cv2.destroyAllWindows()

        ############## CALIBRATION #######################################################

        ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None,
                                                                    None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        # pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
        # pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
        # pickle.dump(dist, open( "dist.pkl", "wb" ))
        if ret:
            print(f'camera_matrix: {cameraMatrix}\ndist: {dist}')
            np.savez(
                f"{self.calib_data_path}/MultiMatrix_{gray_image.shape[0]}",
                camMatrix=cameraMatrix,
                distCoef=dist,
                rVector=rvecs,
                tVector=tvecs,
            )

            # Reprojection Error
            mean_error = 0

            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error

            print("total error: {}".format(mean_error / len(objpoints)))
        return ret

    def calibrate_charuco(self):
        corners, ids, size = self.read_chessboards()
        return self.calibrate_camera(corners, ids, size)

    def read_chessboards(self):
        """
        Charuco base pose estimation.
        """
        print("POSE ESTIMATION STARTS:")
        allCorners = []
        allIds = []
        decimator = 0
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        for im in self.saved_images:
            # print("=> Processing image {0}".format(im))
            # frame = cv2.imread(im)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.aruco_dict)

            if len(corners) > 0:
                # SUB PIXEL DETECTION
                for corner in corners:
                    cv2.cornerSubPix(gray, corner,
                                     winSize=(3, 3),
                                     zeroZone=(-1, -1),
                                     criteria=criteria)
                res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
                if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])

            decimator += 1

        imsize = gray.shape
        return allCorners, allIds, imsize

    def calibrate_camera(self, allCorners, allIds, imsize):
        """
        Calibrates the camera using the dected corners.
        """
        print("CAMERA CALIBRATION")

        cameraMatrixInit = np.array([[1400., 0., imsize[0] / 2.],
                                     [0., 1400., imsize[1] / 2.],
                                     [0., 0., 1.]])

        distCoeffsInit = np.zeros((5, 1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
        # # flags = (cv2.CALIB_RATIONAL_MODEL)
        # (ret, camera_matrix, distortion_coefficients0,
        #  rotation_vectors, translation_vectors,
        #  stdDeviationsIntrinsics, stdDeviationsExtrinsics,
        #  perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        #     charucoCorners=allCorners,
        #     charucoIds=allIds,
        #     board=self.board,
        #     imageSize=imsize,
        #     cameraMatrix=cameraMatrixInit,
        #     distCoeffs=distCoeffsInit,
        #     flags=flags,
        #     criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        calibrations = self.solve_all_sample_combos(allCorners=allCorners, allIds=allIds, imsize=imsize,
                                                    cameraMatrixInit=cameraMatrixInit, distCoeffsInit=distCoeffsInit,
                                                    flags=flags, criteria=self.criteria)

        print(calibrations)

        # (ret, camera_matrix, distortion_coefficients0,
        #  rotation_vectors, translation_vectors) = cv2.aruco.calibrateCameraCharuco(
        #     charucoCorners=allCorners,
        #     charucoIds=allIds,
        #     board=self.board,
        #     imageSize=imsize,
        #     cameraMatrix=cameraMatrixInit,
        #     distCoeffs=distCoeffsInit,
        #     flags=flags,
        #     criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 30, 1e-9))
        # print(ret)
        # if ret:
        #     print(f'camera_matrix: \n{camera_matrix}\ndist: \n{distortion_coefficients0}')
        #     np.savez(
        #         f"{self.calib_data_path}/MultiMatrix_{imsize}",
        #         camMatrix=camera_matrix,
        #         distCoef=distortion_coefficients0,
        #         rVector=rotation_vectors,
        #         tVector=translation_vectors,
        #     )

        # # Reprojection Error
        # mean_error = 0
        #
        # for i in range(len(allCorners)):
        #     imgpoints2, _ = cv2.projectPoints(allCorners[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        #     mean_error += error
        #
        # print("total error: {}".format(mean_error / len(objpoints)))
        #
        # print(translation_vectors)
        return calibrations

        # return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

    def solve_all_sample_combos(self, allCorners, allIds, imsize, cameraMatrixInit, distCoeffsInit, flags, criteria):

        # Generate all combinations of 3 to n samples
        num_samples_range = range(3, 11)
        all_combinations = []

        calibrations = dict()

        for r in num_samples_range:
            combinations_r = combinations(zip(allIds, allCorners), r)
            all_combinations.extend(combinations_r)

        # Iterate over all combinations
        num_iterations_for_len = 0
        previous_iteration_len = 0
        for combination in all_combinations:

            # Extract corners and ids from the combination
            selected_ids, selected_corners = zip(*combination)

            if len(selected_corners) == previous_iteration_len:
                num_iterations_for_len += 1
            else:
                num_iterations_for_len = 0
                previous_iteration_len = len(selected_corners)

            if num_iterations_for_len >= 100:
                continue
            if len(selected_corners) not in calibrations.keys():
                calibrations[len(selected_corners)] = []
            print(len(selected_corners))
            # Perform camera calibration using selected corners and ids
            ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                selected_corners, selected_ids, self.board, imsize,
                cameraMatrix=cameraMatrixInit, distCoeffs=distCoeffsInit,
                rvecs=None, tvecs=None, criteria=criteria
            )
            calibrations[len(selected_corners)].append(ret)
            # print(calibrations)
        return calibrations
        #
        # step_size = 1
        # start_sample_size = 3
        # end_sample_size = len(self.saved_images) + 1
        # # end_sample_size = 29
        #
        # calibrations = dict()
        # list_size = len(self.saved_images)
        #
        # max_iterations = 0
        #
        # # For every sample size
        # for sample_size in range(start_sample_size, end_sample_size, step_size):
        #     # print(sample_size)
        #
        #     calibrations[sample_size] = list()
        #
        #     # For every index combination
        #     for sample_indices in combinations(range(list_size), sample_size):
        #         # Take out subset of indices
        #         allCornersSubset = [allCorners[index] for index in sample_indices]
        #         allIdsSubset = [allIds[index] for index in sample_indices]
        #         allCornersSubset = np.array(allCornersSubset)
        #
        #         (ret, camera_matrix, distortion_coefficients,
        #          rotation_vectors, translation_vectors) = cv2.aruco.calibrateCameraCharuco(
        #             charucoCorners=allCornersSubset,
        #             charucoIds=allIdsSubset,
        #             board=self.board,
        #             imageSize=imsize,
        #             cameraMatrix=cameraMatrixInit,
        #             distCoeffs=distCoeffsInit,
        #             flags=flags,
        #             criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 30, 1e-9))
        #
        #         # Do and save estimation
        #         calibrations[sample_size].append(
        #             ret
        #         )
        #     print(calibrations)
        #     #     max_iterations += 1
        #     #     if max_iterations >= 500:
        #     #         break
        #     # max_iterations = 0

        # return calibrations


def main():
    CHESSBOARD_DIM = (7, 9)
    rospy.init_node('internal_calibration_node')
    internal_calibrator = InternalCalibrator(board_dimensions=CHESSBOARD_DIM, square_length=0.015, marker_length=0.012)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
