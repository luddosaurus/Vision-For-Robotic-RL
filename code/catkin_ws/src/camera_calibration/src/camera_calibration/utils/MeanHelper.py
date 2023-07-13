# Standard

import numpy as np
from scipy.spatial.transform import Rotation

# OpenCV
import cv2

# TF
import tf.transformations as tf
from camera_calibration.utils.TypeConverter import TypeConverter


class MeanHelper:

    @staticmethod
    def riemannian_mean(transformations, clean_translation=True, clean_rotation=True):
        translations = list()
        rotations = list()
        if type(transformations) is dict:
            transformations = [value for value in transformations.values()]
        for stamped_transform in transformations:
            transform = stamped_transform.transform

            rotations.append(
                np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]))
            translations.append(
                np.array([stamped_transform.transform.translation.x,
                          stamped_transform.transform.translation.y,
                          stamped_transform.transform.translation.z]))
        # print(len(translations))

        clean_translations, clean_rotations = MeanHelper.remove_outliers(np.array(translations), np.array(rotations),
                                                                         clean_translation, clean_rotation)

        return MeanHelper.riemannian_mean_translation(clean_translations), MeanHelper.riemannian_mean_rotation(
            clean_rotations)

    @staticmethod
    def riemannian_mean_translation(translations):
        # Compute the Riemannian mean of the vectors
        # print_t = np.array(translations)

        mean_vector = np.mean(translations, axis=0)
        prev_mean_vector = np.zeros_like(mean_vector)
        while np.linalg.norm(mean_vector - prev_mean_vector) > 1e-6:
            weights = 1 / np.linalg.norm(translations, axis=1)
            mean_vector = np.average(translations, axis=0, weights=weights)
            prev_mean_vector = mean_vector

        # Print the results
        # print("Mean vector:", mean_vector)
        return mean_vector

    @staticmethod
    def riemannian_mean_rotation(quaternions):


        # Convert the quaternion vectors to Rotation objects
        rotations = Rotation.from_quat(quaternions)

        # Compute the Riemannian mean of the rotation quaternions
        mean_rotation = rotations.mean().as_quat()

        # Convert the mean rotation vector back to a quaternion
        mean_quaternion = Rotation.from_quat(mean_rotation)

        # Print the results
        # print("Mean quaternion:", mean_quaternion.as_quat())
        return mean_quaternion.as_quat()

    @staticmethod
    def remove_outliers(translational_vectors_input, rotational_vectors_input, clean_translation, clean_rotation, threshold=1.5):

        translational_vectors = []
        rotational_vectors = []
        for translation, rotation in zip(translational_vectors_input, rotational_vectors_input):
            if not np.isnan(translation).any() and not np.isnan(rotation).any():
                translational_vectors.append(translation)
                rotational_vectors.append(rotation)

        # print(f'NEW INSTANCE------------------------------------{rotational_vectors}')
        # might be problem here
        # if translational_vectors.shape[0] == 0 or rotational_vectors.shape[0] == 0:
        #     print("so triggered :D")
        #     return translational_vectors, rotational_vectors
        # Compute the mean and standard deviation of the two lists
        rotational_vectors_mean = np.mean(rotational_vectors, axis=0)
        translational_vectors_mean = np.mean(translational_vectors, axis=0)

        rotational_vectors_std = np.std(rotational_vectors, axis=0)
        translational_vectors_std = np.std(translational_vectors, axis=0)


        # Compute the Z-scores of each element in the two lists

        if not np.isin(translational_vectors_std, 0).any() or not np.isin(rotational_vectors_std, 0).any():
            rotational_vectors_zscores = np.abs((rotational_vectors - rotational_vectors_mean) / rotational_vectors_std)
            translational_vectors_zscores = np.abs(
                (translational_vectors - translational_vectors_mean) / translational_vectors_std)
            # print((rotational_vectors - rotational_vectors_mean) / rotational_vectors_std)

            # print(translational_vectors_zscores)
            # print('--------------')
            # print(rotational_vectors_zscores)
        else:
            return translational_vectors, rotational_vectors

        # plot_rotational_vectors_zscore(rotational_vectors_zscores)
        to_remove = list()
        for i, (entry_r, entry_t) in enumerate(zip(rotational_vectors_zscores, translational_vectors_zscores)):
            for value_r, value_t in zip(entry_r, entry_t):
                if value_r > threshold or value_t > threshold:
                    to_remove.append(i)

        # # Remove the elements with a Z-score greater than the threshold
        # rotational_vectors_indices = np.where(rotational_vectors_zscores > threshold)
        # translational_vectors_indices = np.where(translational_vectors_zscores > threshold)
        # # print(rotational_vectors_indices, " - ", len(rotational_vectors_indices))
        # # print("--------------------------------------------------------------")
        # # print(translational_vectors_indices, " - ", len(translational_vectors_indices))
        #
        # set1 = set(np.array(rotational_vectors_indices).flatten())
        # set2 = set(np.array(translational_vectors_indices).flatten())
        # indices = np.array(list(set1.union(set2)))
        # print(indices)

        # if len()
        # indices = rotational_vectors_indices.union(translational_vectors_indices)
        if len(to_remove) != 0:
            rotational_vectors_clean = np.delete(rotational_vectors, to_remove, axis=0)
            translational_vectors_clean = np.delete(translational_vectors, to_remove, axis=0)

            return translational_vectors_clean, rotational_vectors_clean

        return translational_vectors, rotational_vectors
