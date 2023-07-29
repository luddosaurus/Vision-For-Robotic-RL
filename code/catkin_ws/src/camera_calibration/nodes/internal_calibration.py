#! /usr/bin/env python3.8


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg

import cv2
import os
import numpy as np

from random import shuffle

from camera_calibration.utils.ARHelper import ARHelper
from camera_calibration.utils.extract_realsense_parameters import ExtractParameters
from camera_calibration.utils.JSONHelper import JSONHelper
from camera_calibration.params.aruco_dicts import ARUCO_DICT
from camera_calibration.utils.DaVinci import DaVinci
from camera_calibration.params.calibration import extrinsic_calibration_results_path
from camera_calibration.utils.HarryPlotterAndTheChamberOfSeaborn import HarryPlotter

CALIB_DATA_NAME = "calib_data_1280"


class InternalCalibrator(object):
    def __init__(self, camera_name=None, factory_settings=None, board_name=None, image_topic=None, save_directory=None):

        # Setup board and charuco data
        self.type = None
        self.board_dimensions = None
        self.square_size = None
        self.aruco_marker_size = None
        self.aruco_dict = None
        self.board = None
        self.arHelper = None

        # Initialize above values depending on board type
        board_parameters = JSONHelper.get_board_parameters(board_name)
        self.setup_board(board_parameters)

        self.factory_settings = factory_settings

        # Setup rospy
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber(image_topic, Image, self.my_callback)

        # As defined by opencv
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # For saving the result
        self.camera_name = camera_name

        # Camera factory parameters for initial guess

        self.saved_images = list()
        self.calibration_results = None
        self.save_directory = save_directory
        self.plot_images = list()

        self.copy_originals = []

    def setup_board(self, board_parameters):
        self.type = board_parameters['type']
        self.board_dimensions = (
            board_parameters['board_shape'][0], board_parameters['board_shape'][1])
        self.square_size = board_parameters['square_size']

        if self.type == 'charuco':
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[board_parameters['dict_type']])
            self.aruco_marker_size = board_parameters['aruco_marker_size']
            self.board = cv2.aruco.CharucoBoard_create(squaresX=self.board_dimensions[1],
                                                       squaresY=self.board_dimensions[0],
                                                       squareLength=self.square_size,
                                                       markerLength=self.aruco_marker_size,
                                                       dictionary=self.aruco_dict)
            self.arHelper = ARHelper(charuco_board_shape=self.board_dimensions,
                                     charuco_square_size=self.square_size,
                                     charuco_marker_size=self.aruco_marker_size,
                                     dict_type=self.aruco_dict)

    def my_callback(self, image_message):

        try:
            image = self.bridge.imgmsg_to_cv2(img_msg=image_message, desired_encoding='bgr8')
        except CvBridgeError:
            print(CvBridgeError)
        original_image = image.copy()

        if self.type == 'checkerboard':
            board_detected, image = self.detect_checkerboard(image=image)
        elif self.type == 'charuco':
            board_detected, image = self.arHelper.detect_and_draw_charuco(image=image)

        # ---------------------- GUI
        info = "[q]uit " \
               "[s]ave " \
               "[u]ndo " \
               "[r]un " \
               "[c]ollect " \
               "[l]oad"
        DaVinci.draw_text_box_in_corner(
            image=image,
            text=info,
            position="bottom_left",
            thickness=1,
            font_scale=0.8
        )
        DaVinci.draw_text_box_in_corner(
            image=image,
            text=f'Number of images captured: {len(self.saved_images)}',
            position='top_left', thickness=1, font_scale=0.8
        )

        if self.calibration_results is not None:
            text = "Reprojection Error: :.3f".format(self.calibration_results[0])
            DaVinci.draw_text_box_in_corner(image=image, text=f'Reprojection Error: {self.calibration_results[0]:.3f}',
                                            position='top_right', thickness=1, font_scale=0.8)
        resized_image = DaVinci.pad_image_cv(image)  # DaVinci.resize(image.copy())

        cv2.imshow('image subscriber', resized_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            rospy.signal_shutdown('We are done here')
        elif key == ord('c') and board_detected:
            self.saved_images.append(original_image)
            # self.run_calibration()
        elif key == ord('r') and len(self.saved_images) >= 1:
            self.run_calibration()
            print(
                f'Camera matrix:\n{self.calibration_results[1]}\nDistortion:\n{self.calibration_results[2].flatten()}')
        elif key == ord('s'):
            if self.calibration_results is not None:
                JSONHelper.save_intrinsics(camera_name=self.camera_name, camera_matrix=self.calibration_results[1],
                                           distortion=self.calibration_results[2].flatten(),
                                           image_shape=original_image.shape)
            if self.save_directory is not None:
                self.save_images()

        elif key == ord('u') and len(self.saved_images) >= 1:
            self.saved_images = self.saved_images[:-1]
        elif key == ord('l') and self.save_directory is not None:
            self.load_images()
        elif key == ord('p'):
            self.saved_images = self.copy_originals.copy()
            plot_images = self.saved_images.copy()
            self.saved_images = []
            calib_results = []
            for im in plot_images:
                self.saved_images.append(im)
                self.run_calibration()
                calib_results.append(self.calibration_results[1])
            print('plotting!')
            HarryPlotter.plot_scatter_image_count(calib_results, plot_images[0].shape)
        elif key == ord('o'):
            self.saved_images = self.copy_originals.copy()
            plot_images = self.saved_images.copy()
            self.saved_images = []
            calib_results_1 = []
            calib_results_2 = []
            for im in plot_images:
                self.saved_images.append(im)
                self.run_calibration()
                calib_results_1.append(self.calibration_results)
            self.factory_settings = None
            self.saved_images = []
            for im in plot_images:
                self.saved_images.append(im)
                self.run_calibration()
                calib_results_2.append(self.calibration_results)
            print('plotting!')
            HarryPlotter.plot_intrinsic_guess(calib_results_1, calib_results_2, 'With initial guess',
                                              'No initial guess')
        elif key == ord('i'):
            self.saved_images = self.copy_originals.copy()
            # shuffle(self.saved_images)
            plot_images = self.saved_images.copy()
            self.saved_images = []
            calib_results_far = []
            calib_results_near = []
            for im in plot_images[:int(len(plot_images) / 2)]:
                self.saved_images.append(im)
                self.run_calibration()
                calib_results_far.append(self.calibration_results)

            self.saved_images = []
            for im in plot_images[int(len(plot_images) / 2):]:
                self.saved_images.append(im)
                self.run_calibration()
                calib_results_near.append(self.calibration_results)
            print('plotting!')
            HarryPlotter.plot_intrinsic_guess(calib_results_far, calib_results_near, 'Far away', 'Near')

    def run_calibration(self):
        if self.type == 'checkerboard':
            self.calibrate()
        else:
            self.calibrate_charuco()

    def calibrate_charuco(self):
        corners, ids, size = self.find_charuco_corners()
        return self.calibrate_camera(corners, ids, size)

    def save_images(self):
        save_path = extrinsic_calibration_results_path + 'intrinsic/' + self.save_directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, image in enumerate(self.saved_images):
            image_path = os.path.join(save_path, f'image_{i}.png')
            cv2.imwrite(image_path, image)

    def load_images(self):
        save_path = extrinsic_calibration_results_path + 'intrinsic/' + self.save_directory
        if not os.path.exists(save_path):
            print(f"Error: No dir {save_path}")
            return

        files = os.listdir(save_path)

        image_files = [file for file in files if file.endswith(".png")]

        for image_file in image_files:
            image_path = os.path.join(save_path, image_file)
            image = cv2.imread(image_path)

            if image is not None:
                self.saved_images.append(image)
            else:
                print(f"No image {image_file}")
        self.copy_originals = self.saved_images.copy()

    def find_charuco_corners(self):
        """
        Charuco base pose estimation.
        """
        print("POSE ESTIMATION STARTS:")
        all_corners = []
        all_ids = []
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
                    all_corners.append(res2[1])
                    all_ids.append(res2[2])

            decimator += 1

        imsize = gray.shape
        return all_corners, all_ids, imsize

    def calibrate_camera(self, all_corners, all_ids, image_size):

        camera_matrix_init = None
        dist_coeffs_init = None
        flags = 0
        if self.factory_settings is not None:
            camera_matrix_init = np.array(
                [[self.factory_settings['focal_point'][0], 0., self.factory_settings['center_point'][0]],
                 [0., self.factory_settings['focal_point'][1], self.factory_settings['center_point'][1]],
                 [0., 0., 1.]])

            dist_coeffs_init = np.array(self.factory_settings['distortion']).transpose()
            flags = cv2.CALIB_USE_INTRINSIC_GUESS
            # dist_copy = dist_coeffs_init.copy()
        # cameraMatrixInit = np.array([[1387., 0., 946],
        #                              [0., 1388., 561],
        #                              [0., 0., 1.]])
        #
        # distCoeffsInit = np.zeros((5, 1))

        (reprojection_error, camera_matrix, distortion, rotation_vectors,
         translation_vectors) = cv2.aruco.calibrateCameraCharuco(charucoCorners=all_corners, charucoIds=all_ids,
                                                                 board=self.board, imageSize=image_size,
                                                                 cameraMatrix=camera_matrix_init,
                                                                 distCoeffs=dist_coeffs_init, flags=flags, criteria=(
                cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        # (reprojection_error, camera_matrix, distortion,
        #  rotation_vectors, translation_vectors,
        #  stdDeviationsIntrinsics, stdDeviationsExtrinsics,
        #  perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        #     charucoCorners=all_corners,
        #     charucoIds=all_ids,
        #     board=self.board,
        #     imageSize=image_size,
        #     cameraMatrix=camera_matrix_init,
        #     distCoeffs=dist_coeffs_init,
        #     flags=flags,
        #     criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        self.calibration_results = (reprojection_error, camera_matrix, distortion)

    ######################################################## CHECKERBOARD CALIBRATION ##############################################################

    def detect_checkerboard(self, image):
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_scale_image, self.board_dimensions, None)

        if ret:
            corners_sub = cv2.cornerSubPix(gray_scale_image, corners, (3, 3), (-1, -1), self.criteria)
            image = cv2.drawChessboardCorners(image, self.board_dimensions, corners_sub, ret)

        return ret, image

    def calibrate(self):
        camera_matrix_init = None
        dist_coeffs_init = None
        flags = 0
        if self.factory_settings is not None:
            camera_matrix_init = np.array(
                [[self.factory_settings['focal_point'][0], 0., self.factory_settings['center_point'][0]],
                 [0., self.factory_settings['focal_point'][1], self.factory_settings['center_point'][1]],
                 [0., 0., 1.]])

            # distCoeffsInit = np.zeros((5, 1))
            dist_coeffs_init = np.array(self.factory_settings['distortion'])
            flags = cv2.CALIB_USE_INTRINSIC_GUESS

        objp = np.zeros((self.board_dimensions[0] * self.board_dimensions[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_dimensions[0], 0:self.board_dimensions[1]].T.reshape(-1, 2)

        objp = objp * self.square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        num_used_images = 0

        for calibration_image in self.saved_images:

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

        cv2.destroyAllWindows()

        ############## CALIBRATION #######################################################

        reprojection_error, camera_matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(objectPoints=objpoints,
                                                                                            imagePoints=imgpoints,
                                                                                            imageSize=gray_image.shape[
                                                                                                      ::-1],
                                                                                            cameraMatrix=camera_matrix_init,
                                                                                            distCoeffs=dist_coeffs_init,
                                                                                            flags=flags)

        self.calibration_results = (reprojection_error, camera_matrix, distortion)

        return reprojection_error


def main():
    config_file = rospy.get_param(param_name='internal_calibration_node/config')

    camera_name, factory_settings, board_name, image_topic, save_directory = JSONHelper.get_internal_calibration_parameters(
        config_file)

    rospy.init_node('internal_calibration_node')
    internal_calibrator = InternalCalibrator(camera_name=camera_name, factory_settings=factory_settings,
                                             board_name=board_name, image_topic=image_topic,
                                             save_directory=save_directory)
    # internal_calibrator = InternalCalibrator(charuco_board_shape=CHESSBOARD_DIM, charuco_marker_size=0.31,
    #                                          charuco_square_size=0.4, dict_type=cv2.aruco.DICT_5X5_100)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
