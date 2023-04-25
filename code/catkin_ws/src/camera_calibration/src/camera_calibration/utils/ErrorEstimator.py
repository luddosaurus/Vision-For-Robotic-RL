
import numpy as np

class ErrorEstimator:

    def euclidean_distance(transform1, transform2):
        # Convert the transformation matrices to homogeneous coordinates
        homog1 = np.vstack((transform1, [0, 0, 0, 1]))
        homog2 = np.vstack((transform2, [0, 0, 0, 1]))

        # Compute the Euclidean distance between the homogeneous coordinates
        distance = np.linalg.norm(homog1 - homog2)

        return distance