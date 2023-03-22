from cv2 import aruco as aruco

from cv.utils.image_transform import *


class ARHelper:

    # ArUco Marker Settings
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    param_markers = aruco.DetectorParameters()

    marker_size = 50

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

