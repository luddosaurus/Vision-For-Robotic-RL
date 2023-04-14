import cv2
import numpy as np

# Accuracy measures how often the detector correctly identifies a marker 
# Precision measures how often the detector produces a true positive result (correct detection)
def measure_detection_performance(detected_markers, ground_truth_markers):
    detected_set = set(detected_markers.flatten())
    ground_truth_set = set(ground_truth_markers.flatten())

    # Calculate the accuracy (ratio of correctly detected markers to total markers)
    accuracy = len(detected_set.intersection(ground_truth_set)) / len(ground_truth_set)

    # Calculate the precision (ratio of true positives to all detected markers)
    true_positives = len(detected_set.intersection(ground_truth_set))
    false_positives = len(detected_set.difference(ground_truth_set))
    precision = true_positives / (true_positives + false_positives)

    return accuracy, precision
