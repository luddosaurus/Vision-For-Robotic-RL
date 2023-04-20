import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import lsqr
from camera_calibration.utils.TFTransformer import TFTransformer
import cv2


def estimate_camera_position(aruco_to_camera_transforms):
    transform_matrices \
        = TFTransformer.stamped_transforms_to_matrices(aruco_to_camera_transforms)
    camera_position = lsqr_estimate(transform_matrices)


def lsqr_estimate(transform_matrices):
    # (translation, rotation)
    A = np.zeros((3 * len(transform_matrices), 6))
    B = np.zeros((3 * len(transform_matrices), 1))

    # Fill in the A and B matrices
    for i, matrix in enumerate(transform_matrices):
        r = matrix[:3, :3]
        t = matrix[:3, 3]

        A[i*3:i*3+3, :] = np.hstack((r, np.eye(3)))
        B[i*3:i*3+3, :] = t.reshape(3, 1)

    # Solve the linear system using lsqr
    x, _, _, _ = lsqr(A, B)

    # Extract the rotation and translation from the solution vector
    r = x[:3]
    t = x[3:]

    # Construct the transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(r)[0]
    matrix[:3, 3] = t.reshape(3)

    return matrix


def svd_estimate(transform_matrices):
    # Stack the matrices into a 3D array
    A = np.stack(transform_matrices)

    # Extract the rotation and translation components from the matrices
    R = A[:, :3, :3]
    T = A[:, :3, 3]

    # Compute the centroid of the translations
    T_mean = np.mean(T, axis=0)

    # Center the translations around the centroid
    T_centered = T - T_mean

    # Compute the SVD of the rotation matrices
    U, s, Vt = np.linalg.svd(R)

    # Compute the rotation matrix that minimizes the sum of squared distances
    # to the other rotation matrices
    R_mean = U @ Vt

    # Combine the rotation and translation components into a single transformation matrix
    A_mean = np.zeros((4, 4))
    A_mean[:3, :3] = R_mean
    A_mean[:3, 3] = T_mean
    A_mean[3, 3] = 1

    # Extract the position component from the transformation matrix
    position = A_mean[:3, 3]

    return position

