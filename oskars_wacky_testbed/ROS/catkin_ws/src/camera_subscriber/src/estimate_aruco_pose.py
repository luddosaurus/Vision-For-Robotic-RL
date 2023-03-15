from aruco_marker_set import MarkerSet
import cv2


def estimate_pose(image, aruco_dict_type, matrix_coefficients, distortion_coefficients, marker_size):
    gray_scale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray_scale_image, cv2.aruco_dict, parameters=parameters)

    distances = dict()
    centers = dict()

    if len(corners) > 0:
        marker_set = MarkerSet(corners, ids, marker_size, matrix_coefficients, distortion_coefficients, image)
        image = marker_set.draw_markers_with_axes()
        distances = marker_set.get_camera_distance_to_markers_via_transform()
        centers = marker_set.get_center_of_every_marker()

    return image, distances, centers

