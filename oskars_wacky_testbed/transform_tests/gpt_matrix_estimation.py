import numpy as np
from scipy.spatial import KDTree


def icp(source_points, target_points, max_iterations=50, tolerance=1e-6):
    """
    Iterative Closest Point (ICP) algorithm for point cloud registration.

    Parameters:
    source_points (numpy array): Point cloud to be registered.
    target_points (numpy array): Reference point cloud.
    max_iterations (int): Maximum number of iterations for the algorithm.
    tolerance (float): Tolerance for stopping the algorithm.

    Returns:
    transformation_matrix (numpy array): 4x4 transformation matrix that aligns the source points to the target points.
    """

    # Initialize the transformation matrix to the identity matrix
    transformation_matrix = np.eye(4)
    print(transformation_matrix.shape)

    # Loop for the maximum number of iterations
    for i in range(max_iterations):
        # Transform the source points using the current transformation matrix
        ones = np.ones((len(source_points), 1))
        test = np.vstack((source_points.reshape((4, 1)), ones))
        print(test.shape)
        transformed_points = transformation_matrix.dot(test)

        np.dot()

        # Find the nearest neighbors of each transformed source point in the target point cloud
        tree = KDTree(target_points)
        distances, indices = tree.query(transformed_points)

        # Compute the mean square error
        mse = np.mean(distances ** 2)

        # Check if the algorithm has converged
        if mse < tolerance:
            break

        # Compute the centroids of the source and target point clouds
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        # Compute the cross-covariance matrix
        cross_covariance = (source_points - source_centroid).T.dot(target_points - target_centroid)

        # Compute the singular value decomposition of the cross-covariance matrix
        u, s, vh = np.linalg.svd(cross_covariance)

        # Construct the rotation matrix and translation vector
        rotation_matrix = vh.T.dot(u.T)
        translation_vector = target_centroid - rotation_matrix.dot(source_centroid)

        # Construct the transformation matrix
        new_transformation_matrix = np.eye(4)
        new_transformation_matrix[:3, :3] = rotation_matrix
        new_transformation_matrix[:3, 3] = translation_vector

        # Update the current transformation matrix
        transformation_matrix = new_transformation_matrix.dot(transformation_matrix)

    return transformation_matrix


point1 = np.asarray([1, 2, 3, 4])
point2 = np.asarray([5, 6, 7, 8])

print(icp(point1, point2))
