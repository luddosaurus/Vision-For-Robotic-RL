#! /usr/bin/python3.8

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg

import cv2
import os
import numpy as np

CALIB_DATA_NAME = "calib_data_1280"


class InternalCalibrator(object):
    def __init__(self, board_dimensions, square_size_mm):
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.my_callback)
        self.board_dimensions = board_dimensions
        self.size_of_chessboard_squares_mm = square_size_mm
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.saved_images = list()
        self.calib_data_path = os.path.join(rospkg.RosPack().get_path('camera_calibration'),
                                            'calibration_data/internal_calibration')

    def my_callback(self, image_message):

        try:
            image = self.bridge.imgmsg_to_cv2(img_msg=image_message, desired_encoding='bgr8')
        except CvBridgeError:
            print(CvBridgeError)
        original_image = image.copy()
        _, board_detected = self.detect_checkerboard(image=image)

        if len(self.saved_images) < 5:
            cv2.imshow('image subscriber', image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            rospy.signal_shutdown('We are done here')
        elif key == ord('s') and board_detected:
            self.saved_images.append(original_image)
        calibration_finished = False
        if len(self.saved_images) >= 5:
            calibration_finished = self.calibrate()
        if calibration_finished:
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
                f"{self.calib_data_path}/MultiMatrix_testing_new_node{gray_image.shape[0]}",
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
    CHESSBOARD_DIM = (7, 10)
    rospy.init_node('internal_calibration_node')
    internal_calibrator = InternalCalibrator(board_dimensions=CHESSBOARD_DIM, square_size_mm=15)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
