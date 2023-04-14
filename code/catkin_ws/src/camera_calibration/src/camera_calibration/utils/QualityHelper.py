import numpy as np
import cv2

class QualityHelper:

    def __init__(self, marker_ids):
        self.marker_ids = marker_ids
        self.detected_marker_ids_count = {key: 0 for key in marker_ids}
        self.frame_count = 0
        self.precision_history = list()
        self.accuracy_history = list()

    def get_detection_rate(self):
        detection_rate = []
        for key in self.detected_marker_ids_count:
            detection_rate[key] = self.detected_marker_ids_count[key] / self.frame_count

    def detected(self, detected_markers):
        self.frame_count += 1
        for item in detected_markers:
            if item in self.detected_marker_ids_count:
                self.detected_marker_ids_count[item] += 1

    # Accuracy measures how often the detector correctly identifies a marker
    # Precision measures how often the detector produces a true positive result (correct detection)
    def measure_detection_performance(self, detected_markers):
        detected_set = set(detected_markers.flatten())
        ground_truth_set = set(self.marker_ids.flatten())
        # Calculate the accuracy (ratio of correctly detected markers to total markers)
        accuracy = len(detected_set.intersection(ground_truth_set)) / len(ground_truth_set)

        # Calculate the precision (ratio of true positives to all detected markers)
        true_positives = len(detected_set.intersection(ground_truth_set))
        false_positives = len(detected_set.difference(ground_truth_set))
        precision = true_positives / (true_positives + false_positives)
        self.accuracy_history.append(accuracy)
        self.precision_history.append(precision)
        return accuracy, precision


# Euclidean distance between the projected 2D image points and their corresponding detected 2D image points.
def calculate_reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):

    projected_points, _ = cv2.projectPoints(obj_points, rvecs, tvecs, camera_matrix, dist_coeffs)
    dists = np.linalg.norm(projected_points.squeeze() - img_points, axis=1)

    # Calculate the mean reprojection error
    mean_reproj_error = np.mean(dists)

    return mean_reproj_error


def measure_pose_consistency(transforms):
    rotations = []
    translations = []
    for transform in transforms:
        rot_quat = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]
        trans_vec = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ]
        rotations.append(rot_quat)
        translations.append(trans_vec)

    rotation_std = np.std(rotations, axis=0)
    translation_std = np.std(translations, axis=0)

    return rotation_std, translation_std
