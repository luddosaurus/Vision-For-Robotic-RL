import os

import numpy as np
import cv2

# todo
#  - choose good images
#  - Distortion checker


class Calibrator:

    # --------------------- SETTINGS

    def __init__(self, size_y=6, size_x=9, square_size_mm=24):

        # Calibration Settings
        self.chessboard_y = size_y
        self.chessboard_x = size_x
        self.chessboard_size = (self.chessboard_x, self.chessboard_y)
        size_of_a_square_mm = square_size_mm

        # termination criteria for finding good corners
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.chessboard_y * self.chessboard_x, 3), np.float32)
        self.objp[:, :2] = (size_of_a_square_mm * np.mgrid[0:self.chessboard_x, 0:self.chessboard_y]).T.reshape(-1, 2)
        # objp[:, :2] = (size_of_a_square * np.mgrid[0:7, 0:6]).T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.obj_points = []  # 3d point in real world space, World points?
        self.img_points = []  # 2d points in image plane.

        # Intrinsic Parameters
        self.distortion_coefficients = []
        self.rotation_vectors = []
        self.translation_vectors = []
        self.camera_matrix = []

        # Extrinsic Matrix
        self.extrinsic_rotation_matrix = []
        self.extrinsic_translation_vector = []

        self.save_directory = "calibrations"

    def reset_settings(self):
        self.obj_points = []
        self.img_points = []
        self.chessboard_y = 5
        self.chessboard_x = 8
        self.chessboard_size = (self.chessboard_x, self.chessboard_y)

    def clear_points(self):
        self.obj_points = []
        self.img_points = []

    # --------------------- CALIBRATION

    def find_chessboard(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        found_pattern, corners = cv2.findChessboardCorners(
            image=gray_img,
            patternSize=self.chessboard_size
        )

        if found_pattern:
            self.obj_points.append(self.objp)
            # find 'clean' corner
            clean_corners = cv2.cornerSubPix(
                image=gray_img,
                corners=corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=self.criteria
            )

            return found_pattern, clean_corners

        else:
            return False, []

    def save_chessboard_points(self, corners):
        self.img_points.append(corners)
        # print("Images : ", len(self.img_points))

    def find_and_save_chessboard_points(self, img):
        found_pattern, corners = self.find_chessboard(img=img)
        if found_pattern:
            self.save_chessboard_points(corners)

        return found_pattern, corners

    def calibrate_extrinsic(self):
        if not self.camera_matrix:
            return Exception("Calibrate intrinsic first :)")

        ret, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=self.obj_points,
            imagePoints=self.img_points,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.distortion_coefficients)

        rot_mat, _ = cv2.Rodrigues(rotation_vector)
        self.extrinsic_translation_vector = translation_vector
        self.extrinsic_rotation_matrix = rot_mat

    def calibrate_intrinsic(self, img, print_parameters=False):
        if not self.obj_points or len(self.obj_points) == 0:
            raise Exception("Find chessboard points first")

        height, width = img.shape[:2]
        image_size = (width, height)

        ret, camera_matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(
            objectPoints=self.obj_points,
            imagePoints=self.img_points,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None)

        if print_parameters:
            print("Camera Calibrated: ", ret)
            print("Camera Matrix: ", camera_matrix)
            print("Camera Distortion: ", distortion)
            print("Camera Rotation: ", rotation_vectors)
            print("Camera Translation: ", translation_vectors)
            print("Object Points: ", len(self.obj_points))

        optimised_matrix, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=camera_matrix,
            distCoeffs=distortion,
            imageSize=image_size,
            alpha=1,
            newImgSize=image_size
        )

        image_undistorted = cv2.undistort(
            src=img,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion,
            dst=None,
            newCameraMatrix=optimised_matrix
        )
        x, y, w, h, = roi
        image_undistorted = image_undistorted[y:y + h, x:x + w]

        self.camera_matrix = optimised_matrix
        self.distortion_coefficients = distortion
        self.rotation_vectors = rotation_vectors
        self.translation_vectors = translation_vectors

        return image_undistorted
    # --------------------- DRAW IMAGES

    def draw_chessboard_on_image(self, img, corners, found_pattern=True):
        cv2.drawChessboardCorners(
            image=img,
            patternSize=self.chessboard_size,
            corners=corners,
            patternWasFound=found_pattern
        )
        return img

    # --------------------- ERROR

    def find_error(self, print_result=False):
        if len(self.obj_points) == 0:
            raise Exception("Find some points first!q")
        # Reprojection Error
        mean_error = 0

        for i in range(len(self.obj_points)):
            projectedPoints, _ = cv2.projectPoints(
                self.obj_points[i],
                self.rotation_vectors[i],
                self.translation_vectors[i],
                self.camera_matrix,
                self.distortion_coefficients
            )
            error = cv2.norm(self.img_points[i], projectedPoints, cv2.NORM_L2) / len(projectedPoints)
            mean_error += error

        if print_result:
            print("Total error: {}".format(mean_error / len(self.obj_points)))
            print("Distortion: {}".format(self.distortion_coefficients))

        return mean_error / len(self.obj_points)

    # --------------------- STORAGE

    def save_calibration(self, save_name="camera_parameters"):
        self.__make_dir()
        e_path = f"{self.save_directory}/{save_name}_extrinsic"
        i_path = f"{self.save_directory}/{save_name}_intrinsic"
        np.savez(
            file=i_path,
            matrix=self.camera_matrix,
            distortion=self.distortion_coefficients,
            rotation_vectors=self.rotation_vectors,
            translation_vectors=self.translation_vectors,
        )
        np.savez(
            file=e_path,
            rotation_matrix=self.extrinsic_rotation_matrix,
            translation_vector=self.extrinsic_translation_vector
        )

    def __make_dir(self):
        save_dir = f"{self.save_directory}"
        CHECK_DIR = os.path.isdir(save_dir)
        if not CHECK_DIR:
            os.makedirs(save_dir)

    def load_calibration(self, path):
        e_path = f"{path}_extrinsic"
        i_path = f"{path}_intrinsic"

        with np.load(f"{i_path}.npz") as X:
            self.camera_matrix, self.distortion_coefficients, self.rotation_vectors, self.translation_vectors = \
                [X[i] for i in ('matrix', 'distortion', 'rotation_vectors', 'translation_vectors')]

        with np.load(f"{e_path}.npz") as X:
            self.camera_matrix, self.extrinsic_rotation_matrix, self.extrinsic_translation_vector = \
                [X[i] for i in ('rotation_matrix', 'translation_vector')]

