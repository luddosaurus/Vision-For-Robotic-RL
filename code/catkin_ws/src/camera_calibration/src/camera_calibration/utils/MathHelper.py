# Standard
import time

import numpy as np
from scipy.spatial.transform import Rotation

# OpenCV
import cv2

# TF
import tf.transformations as tf


# Invert vector from O to (translation, rotation)
def invert_transform(translation, rotation, turn_into_quaternion=True):
    rotation_mat, _ = cv2.Rodrigues(rotation)

    # Change frame from Camera to ArUco, to ArUco to Camera
    inv_rotation = np.transpose(rotation_mat)
    inv_translation = np.matmul(-inv_rotation, translation.T)

    q_normalized = rotation_vector_to_quaternions(rotation)

    if turn_into_quaternion:
        return inv_translation, q_normalized
    else:
        return inv_translation, inv_rotation


def rotation_vector_to_quaternions(rotation_vector):
    # Embed the rotation matrix in a 4x4 transformation matrix for the quaternion
    embedded_rotation = np.eye(4)
    embedded_rotation[:3, :3] = rotation_vector

    # Convert to Quaternion
    quaternion = tf.quaternion_from_matrix(embedded_rotation)

    # Normalize the quaternion because it's important
    q_norm = np.linalg.norm(quaternion)
    q_normalized = quaternion / q_norm

    return q_normalized


def riemannian_mean(transformations):
    translations = list()
    rotations = list()
    for stamped_transform in transformations:
        transform = stamped_transform.transform
        rotations.append(
            np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]))
        translations.append(
            np.array([stamped_transform.transform.translation.x,
                      stamped_transform.transform.translation.y,
                      stamped_transform.transform.translation.z]))
    # print(len(translations))
    clean_translations, clean_rotations = remove_outliers(np.array(translations), np.array(rotations))

    return riemannian_mean_translation(clean_translations), riemannian_mean_rotation(clean_rotations)


def riemannian_mean_translation(translations):
    # Compute the Riemannian mean of the vectors
    mean_vector = np.mean(translations, axis=0)
    prev_mean_vector = np.zeros_like(mean_vector)
    while np.linalg.norm(mean_vector - prev_mean_vector) > 1e-6:
        weights = 1 / np.linalg.norm(translations, axis=1)
        mean_vector = np.average(translations, axis=0, weights=weights)
        prev_mean_vector = mean_vector

    # Print the results
    # print("Mean vector:", mean_vector)
    return mean_vector


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


def remove_outliers(translational_vectors, rotational_vectors, threshold=1):

    # Compute the mean and standard deviation of the two lists
    rotational_vectors_mean = np.mean(rotational_vectors, axis=0)
    translational_vectors_mean = np.mean(translational_vectors, axis=0)

    rotational_vectors_std = np.std(rotational_vectors, axis=0)
    translational_vectors_std = np.std(translational_vectors, axis=0)

    # Compute the Z-scores of each element in the two lists
    rotational_vectors_zscores = np.abs((rotational_vectors - rotational_vectors_mean) / rotational_vectors_std)
    translational_vectors_zscores = np.abs(
        (translational_vectors - translational_vectors_mean) / translational_vectors_std)

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


import matplotlib.pyplot as plt


def plot_rotational_vectors_zscore(rotational_vectors, sc=None):
    zscores = np.abs((rotational_vectors - np.mean(rotational_vectors)) / np.std(rotational_vectors))
    red_green = False
    if red_green:

        if sc is None:
            fig, ax = plt.subplots()
            ax.set_xlabel('Index')
            ax.set_ylabel('Z-score')
            ax.set_title('Z-scores of Rotational Vectors')
            sc = ax.scatter(range(len(rotational_vectors)), np.zeros(len(rotational_vectors)), c='blue')

        # zscores = np.abs((rotational_vectors - np.mean(rotational_vectors)) / np.std(rotational_vectors))
        is_outlier = np.all(zscores < 2, axis=1)
        sc.set_offsets(np.column_stack((range(len(rotational_vectors)), zscores[:, 0])))
        sc.set_color(['red' if o else 'blue' for o in is_outlier])

        plt.pause(1)
        plt.draw()
        time.sleep(2)
        plt.close()
        return sc

        # colors = ['green' if np.any(zscore) < 2 else 'red' for zscore in zscores]
        # print(f'{len(rotational_vectors)}, {len(zscores)}, {len(colors)}')
        # plt.scatter(range(len(rotational_vectors)), zscores, color=colors)
        # plt.xlabel('Index')
        # plt.ylabel('Z-score')
        # plt.title('Z-scores of Rotational Vectors')
        # plt.show()

        # is_outlier = np.all(zscores < 2, axis=1)
        # plt.ion()
        # plt.scatter(range(len(rotational_vectors)), zscores[:, 0], c=zscores[:, 0], cmap='coolwarm')
        # plt.xlabel('Index')
        # plt.ylabel('Z-score')
        # plt.title('Z-scores of Rotational Vectors')
        # plt.colorbar()


    else:

        plt.plot(zscores)
        plt.xlabel('Index')
        plt.ylabel('Z-score')
        plt.title('Z-scores of Rotational Vectors')
        plt.show()
