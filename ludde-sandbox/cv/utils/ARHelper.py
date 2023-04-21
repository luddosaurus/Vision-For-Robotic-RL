from cv2 import aruco as aruco
import cv2

from cv.utils.image_transform import *


class ARHelper(object):
    def __init__(self):
        # ArUco Marker Settings
        marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        param_markers = aruco.DetectorParameters()

        # marker_size = 50

    def find_markers(self, img, debug=False):
        if img.size == 0:
            pass

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get corners (in pixel coordinates) id and rejected
        marker_corners, marker_ids, rejected = aruco.detectMarkers(
            image=gray_image,
            dictionary=self.marker_dict,
            parameters=self.param_markers
        )
        marked_image = aruco.drawDetectedMarkers(image=img, corners=marker_corners, ids=marker_ids)

        if debug:
            print(marker_corners)

        return marked_image, marker_corners, marker_ids

    @staticmethod
    def draw_vectors(self, img, marker_corners, marker_ids, matrix, distortion, marker_size=0.034):

        axis_length = 0.02
        if len(marker_corners) > 0:
            for i in range(0, len(marker_ids)):
                rotation_vec, translation_vec, marker_points = aruco.estimatePoseSingleMarkers(
                    corners=marker_corners[i],
                    markerLength=0.05,
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
