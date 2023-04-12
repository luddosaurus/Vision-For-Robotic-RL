# Standard
import numpy as np

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

    # Embed the rotation matrix in a 4x4 transformation matrix for the quaternion
    embedded_rotation = np.eye(4)
    embedded_rotation[:3, :3] = inv_rotation

    # Convert to Quaternion
    quaternion = tf.quaternion_from_matrix(embedded_rotation)

    # Normalize the quaternion because it's important
    q_norm = np.linalg.norm(quaternion)
    q_normalized = quaternion / q_norm

    if turn_into_quaternion:
        return inv_translation, q_normalized
    else:
        return inv_translation, inv_rotation
