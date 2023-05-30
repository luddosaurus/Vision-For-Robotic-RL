#! /usr/bin/python3.8

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg

import cv2
import os
import numpy as np

from camera_calibration.utils.ARHelper import ARHelper
from camera_calibration.utils.extract_realsense_parameters import ExtractParameters
from camera_calibration.utils.JSONHelper import JSONHelper
from camera_calibration.params.aruco_dicts import ARUCO_DICT

CALIB_DATA_NAME = "calib_data_1280"


class InternalCalibrator(object):
    def __init__(self, camera_name=None, factory_settings=None, board_name=None, image_topic=None):

        # Setup board and charuco data
        self.board_parameters = JSONHelper.get_board_parameters(board_name)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.board_parameters['dict_type']])
        self.board = cv2.aruco.CharucoBoard_create(squaresX=self.board_parameters['charuco_board_shape'][1],
                                                   squaresY=self.board_parameters['charuco_board_shape'][0],
                                                   squareLength=self.board_parameters['charuco_square_size'],
                                                   markerLength=self.board_parameters['charuco_marker_size'],
                                                   dictionary=self.aruco_dict)

        self.board_dimensions = (
            self.board_parameters['charuco_board_shape'][0], self.board_parameters['charuco_board_shape'][1])
        self.arHelper = ARHelper(charuco_board_shape=self.board_dimensions,
                                 charuco_square_size=self.board_parameters['charuco_square_size'],
                                 charuco_marker_size=self.board_parameters['charuco_marker_size'],
                                 dict_type=self.aruco_dict)
        self.size_of_chessboard_squares_mm = self.board_parameters['charuco_square_size']

        # Setup rospy
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber(image_topic, Image, self.my_callback)

        # As defined by opencv
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # For saving the result
        self.camera_name = camera_name

        # Camera factory parameters for initial guess

        self.saved_images = list()

    def my_callback(self, image_message):

        try:
            image = self.bridge.imgmsg_to_cv2(img_msg=image_message, desired_encoding='bgr8')
        except CvBridgeError:
            print(CvBridgeError)
        original_image = image.copy()
        # _, board_detected = self.detect_checkerboard(image=image)
        board_detected, image = self.arHelper.detect_and_draw_charuco(image)

        cv2.imshow('image subscriber', image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            rospy.signal_shutdown('We are done here')
        elif key == ord('c') and board_detected:
            self.saved_images.append(original_image)
        elif key == ord('r') and len(self.saved_images) >= 1:
            calibration_finished = self.calibrate_charuco()

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
        # sub pixel corner detection criterion as defined by opencv
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        for im in self.saved_images:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.aruco_dict)

            if len(corners) > 0:
                for corner in corners:
                    cv2.cornerSubPix(gray, corner,
                                     winSize=(3, 3),
                                     zeroZone=(-1, -1),
                                     criteria=criteria)
                res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)

                if res2[1] is not None and res2[2] is not None and len(res2[1]) > 5 and decimator % 1 == 0:
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

        cameraMatrixInit = np.array([[640., 0., 650],
                                     [0., 639., 401],
                                     [0., 0., 1.]])

        # distCoeffsInit = np.zeros((5, 1))
        distCoeffsInit = np.array([-0.0560283, 0.0681727, -0.000458755, 0.000510362, -0.0216126])

        # cameraMatrixInit = np.array([[1387., 0., 946],
        #                              [0., 1388., 561],
        #                              [0., 0., 1.]])
        #
        # distCoeffsInit = np.zeros((5, 1))
        # flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        # flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_FOCAL_LENGTH + cv2.CALIB_FIX_PRINCIPAL_POINT)
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

        (ret, camera_matrix, distortion_coefficients0,
         rotation_vectors, translation_vectors) = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=allCorners,
            charucoIds=allIds,
            board=self.board,
            imageSize=imsize,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=distCoeffsInit,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
        print(ret)
        if ret:
            print(f'camera_matrix: \n{camera_matrix}\ndist: \n{distortion_coefficients0}')
            np.savez(
                f"{self.calib_data_path}/MultiMatrix_{imsize}",
                camMatrix=camera_matrix,
                distCoef=distortion_coefficients0,
                rVector=rotation_vectors,
                tVector=translation_vectors,
            )

            camera_matrix = np.array([[640.627, 0, 650.553], [0, 639.93, 401.117], [0, 0, 1]])
            distortion_coefficients0 = np.array([-0.0560283, 0.0681727, -0.000458755, 0.000510362, -0.0216126])
            print(f'camera_matrix: \n{camera_matrix}\ndist: \n{distortion_coefficients0}')
            np.savez(
                f"{self.calib_data_path}/d455_default_MultiMatrix_{imsize}",
                camMatrix=camera_matrix,
                distCoef=distortion_coefficients0,
                rVector=rotation_vectors,
                tVector=translation_vectors,
            )

            # Reprojection Error
            mean_error = 0

            # for i in range(len(objpoints)):
            #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            #     mean_error += error
            #
            # print("total error: {}".format(mean_error / len(objpoints)))

            # print(translation_vectors)
        return ret

        # return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

    ######################################################## CHECKERBOARD CALIBRATION ##############################################################

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


def main():
    config_file = rospy.get_param(param_name='internal_calibration_node/config')

    camera_name, factory_settings, board, image_topic = JSONHelper.get_internal_calibration_parameters(config_file)

    rospy.init_node('internal_calibration_node')
    internal_calibrator = InternalCalibrator(camera_name=camera_name, factory_settings=factory_settings,
                                             board_name=board, image_topic=image_topic)
    # internal_calibrator = InternalCalibrator(charuco_board_shape=CHESSBOARD_DIM, charuco_marker_size=0.31,
    #                                          charuco_square_size=0.4, dict_type=cv2.aruco.DICT_5X5_100)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
