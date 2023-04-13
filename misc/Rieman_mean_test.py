from scipy.spatial.transform import Rotation
import numpy as np


def f1():
    # Generate some random translation and rotation vectors
    translations = np.random.rand(10, 3)
    quaternions = np.random.rand(10, 4)

    # Convert the quaternion vectors to Rotation objects
    rotations = Rotation.from_quat(quaternions)

    # Compute the Riemannian mean of the translation and rotation vectors
    mean_translation = np.mean(translations, axis=0)
    mean_rotation = rotations.mean().as_quat()

    # Convert the mean rotation vector back to a quaternion
    mean_quaternion = Rotation.from_quat(mean_rotation)

    # Print the results
    print("Mean translation:", mean_translation)
    print("Mean quaternion:", mean_quaternion.as_quat())


def f2():
    import numpy as np

    # Generate some random translation vectors
    vectors = np.random.rand(10, 3)

    # Compute the Riemannian mean of the vectors
    mean_vector = np.mean(vectors, axis=0)
    prev_mean_vector = np.zeros_like(mean_vector)
    while np.linalg.norm(mean_vector - prev_mean_vector) > 1e-6:
        weights = 1 / np.linalg.norm(vectors, axis=1)
        mean_vector = np.average(vectors, axis=0, weights=weights)
        prev_mean_vector = mean_vector

    # Print the results
    print("Mean vector:", mean_vector)


def f3():
    from scipy.spatial.transform import Rotation
    import numpy as np

    # Generate some random rotation quaternions
    quaternions = np.random.rand(10, 4)

    # Convert the quaternion vectors to Rotation objects
    rotations = Rotation.from_quat(quaternions)

    # Compute the Riemannian mean of the rotation quaternions
    mean_rotation = rotations.mean().as_quat()

    # Convert the mean rotation vector back to a quaternion
    mean_quaternion = Rotation.from_quat(mean_rotation)

    # Print the results
    print("Mean quaternion:", mean_quaternion.as_quat())


f2()
f3()
