import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped


class ErrorEstimator:

    @staticmethod
    def euclidean_distance(transform1, transform2):
        # Convert the transformation matrices to homogeneous coordinates
        homog1 = np.vstack((transform1, [0, 0, 0, 1]))
        homog2 = np.vstack((transform2, [0, 0, 0, 1]))

        # Compute the Euclidean distance between the homogeneous coordinates
        distance = np.linalg.norm(homog1 - homog2)

        return distance

    @staticmethod
    def distance_between(transforms, truth):
        distances = list()
        for stamped_transform in transforms:
            transform = stamped_transform.transform
            distances.append(ErrorEstimator.euclidean_distance(transform, truth))

        return distances

    @staticmethod
    def calculate_stds(transforms):

        num_transforms = len(transforms)
        translations = np.zeros((num_transforms, 3))
        rotations = np.zeros((num_transforms, 4))

        for i, stamped_transform in enumerate(transforms):
            transform = stamped_transform.transform
            translations[i] = [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z
            ]
            rotations[i] = [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w
            ]

        translation_std = np.std(translations, axis=0)
        rotation_std = np.std(rotations, axis=0)

        return translation_std, rotation_std































