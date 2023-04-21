import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
import tf
import numpy as np


class TFTransformer:

    # (4x4) transformation matrix to stamped transform
    @staticmethod
    def matrix_to_stamped_transform(matrix, parent_frame, child_frame):
        stamp = rospy.Time.now()

        translation = Vector3()
        translation.x, translation.y, translation.z = matrix[:3, 3]

        rotation = Quaternion()
        rotation.x, rotation.y, rotation.z, rotation.w \
            = tf.transformations.quaternion_from_matrix(matrix)

        stamped_transform = TransformStamped()
        stamped_transform.header.stamp = stamp
        stamped_transform.header.frame_id = parent_frame
        stamped_transform.child_frame_id = child_frame
        stamped_transform.transform.translation = translation
        stamped_transform.transform.rotation = rotation

        return stamped_transform

    @staticmethod
    def vectors_to_stamped_transform(translation_vector, rotation_vector, parent_frame, child_frame):
        stamp = rospy.Time.now()

        translation = Vector3()

        translation.x, translation.y, translation.z = translation_vector.flatten()  # [0], translation_vector[1], translation_vector[2]
        rotation = Quaternion()
        rotation.x, rotation.y, rotation.z, rotation.w \
            = rotation_vector

        stamped_transform = TransformStamped()
        stamped_transform.header.stamp = stamp
        stamped_transform.header.frame_id = parent_frame
        stamped_transform.child_frame_id = child_frame
        stamped_transform.transform.translation = translation
        stamped_transform.transform.rotation = rotation

        return stamped_transform

    @staticmethod
    def stamped_transforms_to_matrices(stamped_transforms):
        matrices = []
        for stamped_transform in stamped_transforms:
            translation = stamped_transform.transform.translation
            rotation = stamped_transform.transform.rotation

            translation = np.array([translation.x, translation.y, translation.z])
            rotation = np.array([rotation.x, rotation.y, rotation.z, rotation.w])

            # Construct the transformation matrix
            matrix = tf.transformations.compose_matrix(
                translate=translation,
                angles=tf.transformations.euler_from_quaternion(rotation))

            matrices.append(matrix)

        return matrices
