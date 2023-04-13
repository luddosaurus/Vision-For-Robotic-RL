#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy
from camera_calibration.utils.TFPublish import *
import geometry_msgs
from std_msgs.msg import UInt8MultiArray
import tf
import tf2_ros
from tf.transformations import quaternion_matrix
import numpy as np
from time import time


from camera_calibration.params.attached_arucos import table_arucos, arm_arucos
from camera_calibration.utils.MathHelper import riemannian_mean


# def get_numpy_from_transform(transform):
#     translation_array = np.array([transform.translation.x,
#                                   transform.translation.y,
#                                   transform.translation.z])
#     rotation_array = np.array([transform.rotaion.x,
#                                transform.rotaion.y,
#                                transform.rotaion.z,
#                                transform.rotaion.w])
#

# def calculate_average_transform(transforms):
#     """Computes the average transform from a list of TransformStamped messages."""
#     num_transforms = len(transforms)
#
#     # convert transforms to homogeneous transformation matrices
#     matrices = []
#     for stamped_transform in transforms:
#         transform = stamped_transform.transform
#         matrix = quaternion_matrix([transform.rotation.x,
#                                     transform.rotation.y,
#                                     transform.rotation.z,
#                                     transform.rotation.w])
#         # matrix[:3, :3] = np.reshape(transform.transform.rotation.wxyz, (3, 3))
#         matrix[:3, 3] = [stamped_transform.transform.translation.x,
#                          stamped_transform.transform.translation.y,
#                          stamped_transform.transform.translation.z]
#         matrices.append(matrix)
#
#     # compute average homogeneous transformation matrix
#     avg_matrix = np.mean(matrices, axis=0)
#
#     # convert average matrix back to TransformStamped message
#     avg_transform = geometry_msgs.msg.TransformStamped()
#     avg_transform.header.stamp = transforms[0].header.stamp  # use timestamp of first transform
#     avg_transform.header.frame_id = transforms[0].header.frame_id  # use frame ID of first transform
#     avg_transform.child_frame_id = transforms[-1].child_frame_id  # use child frame ID of last transform
#
#     avg_transform.transform.translation.x = avg_matrix[0, 3]
#     avg_transform.transform.translation.y = avg_matrix[1, 3]
#     avg_transform.transform.translation.z = avg_matrix[2, 3]
#     q = tf.transformations.quaternion_from_matrix(avg_matrix)
#     q_norm = np.linalg.norm(q)
#     q = q / q_norm
#     avg_transform.transform.rotation.x = q[0]
#     avg_transform.transform.rotation.y = q[1]
#     avg_transform.transform.rotation.z = q[2]
#     avg_transform.transform.rotation.w = q[3]
#
#     return avg_transform


class StaticCameraPositionEstimator(object):
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
        self.marker_subscriber = rospy.Subscriber('/detected_aruco_marker_ids', UInt8MultiArray, self.marker_callback)
        self.transformations = list()
        self.start_time = time()

    def marker_callback(self, message):
        # marker_ids = message.data
        marker_ids = [int(x) for x in message.data]

        rate = rospy.Rate(10.0)

        transformations = list()

        for aruco in marker_ids:


            # create a tf listener
            # tf_listener = tf.TransformListener()

            # define the source and target frames
            source_frame = 'world'
            target_frame = f'camera_from_aruco_[{aruco}]'

            # define a timeout for the lookup operation
            # timeout = rospy.Duration(1.0)
            # exists = True
            # try:
            #     # check if a transform exists from source_frame to target_frame
            #     exists, _, _ = tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), timeout)
            # except tf.Exception as ex:
            #     # the transform does not exist
            #     print(f"No transform exists from {source_frame} to {target_frame}: {ex}")
            #     continue
            # if exists:
            # if self.tfBuffer.can_transform(target_frame, source_frame, rospy.Time(), timeout):
            try:
                transform = self.tfBuffer.lookup_transform(source_frame, target_frame, rospy.Time())
                self.transformations.append(transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()

        # if len(self.transformations) > 0 and time() - self.start_time > 10:
        if len(self.transformations) > 0:
            self.start_time = time()
            avg_transform = self.create_average_transform_stamped_message(self.transformations)
            publish_static_stamped_transform(publisher=self.pub_aruco_tf,
                                             transform_stamped=avg_transform,
                                             parent_name="world",
                                             child_name="camera_position")
            rospy.sleep(1)
            if len(self.transformations) > 30:
                self.transformations.pop(0)
            # self.transformations = list()
        # publish(trans, pub_aruco_tf)

        # print(trans_from_0)

        rate.sleep()

    def calculate_average_transform(self, transforms):
        """Computes the average transform from a list of TransformStamped messages."""
        num_transforms = len(transforms)

        # convert transforms to homogeneous transformation matrices
        matrices = []
        for stamped_transform in transforms:
            transform = stamped_transform.transform
            matrix = quaternion_matrix([transform.rotation.x,
                                        transform.rotation.y,
                                        transform.rotation.z,
                                        transform.rotation.w])
            # matrix[:3, :3] = np.reshape(transform.transform.rotation.wxyz, (3, 3))
            matrix[:3, 3] = [stamped_transform.transform.translation.x,
                             stamped_transform.transform.translation.y,
                             stamped_transform.transform.translation.z]
            matrices.append(matrix)

        # compute average homogeneous transformation matrix
        avg_matrix = np.mean(matrices, axis=0)

        # convert average matrix back to TransformStamped message
        avg_transform = geometry_msgs.msg.TransformStamped()
        avg_transform.header.stamp = transforms[0].header.stamp  # use timestamp of first transform
        avg_transform.header.frame_id = transforms[0].header.frame_id  # use frame ID of first transform
        avg_transform.child_frame_id = transforms[-1].child_frame_id  # use child frame ID of last transform

        avg_transform.transform.translation.x = avg_matrix[0, 3]
        avg_transform.transform.translation.y = avg_matrix[1, 3]
        avg_transform.transform.translation.z = avg_matrix[2, 3]
        q = tf.transformations.quaternion_from_matrix(avg_matrix)
        q_norm = np.linalg.norm(q)
        q = q / q_norm
        avg_transform.transform.rotation.x = q[0]
        avg_transform.transform.rotation.y = q[1]
        avg_transform.transform.rotation.z = q[2]
        avg_transform.transform.rotation.w = q[3]

        return avg_transform

    def create_average_transform_stamped_message(self, transformations):

        # convert average matrix back to TransformStamped message
        avg_transform = geometry_msgs.msg.TransformStamped()
        avg_transform.header.stamp = transformations[0].header.stamp  # use timestamp of first transform
        avg_transform.header.frame_id = transformations[0].header.frame_id  # use frame ID of first transform
        avg_transform.child_frame_id = transformations[-1].child_frame_id  # use child frame ID of last transform

        mean_translation, mean_rotation = riemannian_mean(transformations)

        avg_transform.transform.translation.x = mean_translation[0]
        avg_transform.transform.translation.y = mean_translation[1]
        avg_transform.transform.translation.z = mean_translation[2]

        avg_transform.transform.rotation.x = mean_rotation[0]
        avg_transform.transform.rotation.y = mean_rotation[1]
        avg_transform.transform.rotation.z = mean_rotation[2]
        avg_transform.transform.rotation.w = mean_rotation[3]

        return avg_transform


# Broadcast Transform: Base -> Camera
if __name__ == '__main__':
    rospy.init_node('external_camera_node')
    static_camera_position_estimator = StaticCameraPositionEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    # tfBuffer = tf2_ros.Buffer()
    # listener = tf2_ros.TransformListener(tfBuffer)
    # pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
    # # pub_aruco_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)
    #
    # rate = rospy.Rate(10.0)
    # while not rospy.is_shutdown():
    #     try:
    #         transformations = list()
    #         aruco_list = arm_arucos + table_arucos
    #         for aruco in aruco_list:
    #             # create a tf listener
    #             tf_listener = tf.TransformListener()
    #
    #             # define the source and target frames
    #             source_frame = 'world'
    #             target_frame = f'camera_from_{aruco.get_name()}'
    #
    #             # define a timeout for the lookup operation
    #             timeout = rospy.Duration(1.0)
    #             # exists = True
    #             # try:
    #             #     # check if a transform exists from source_frame to target_frame
    #             #     exists, _, _ = tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), timeout)
    #             # except tf.Exception as ex:
    #             #     # the transform does not exist
    #             #     print(f"No transform exists from {source_frame} to {target_frame}: {ex}")
    #             #     continue
    #             # if exists:
    #             if tfBuffer.can_transform(target_frame, source_frame, rospy.Time(), timeout):
    #                 transform = tfBuffer.lookup_transform('world', f'camera_from_{aruco.get_name()}', rospy.Time())
    #                 print(transform)
    #                 transformations.append(transform)
    #
    #         avg_transform = calculate_average_transform(transformations)
    #         # publish(trans, pub_aruco_tf)
    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #         rate.sleep()
    #         continue
    #
    #     publish_static_transform_with_transform(publisher=pub_aruco_tf, transform_stamped=avg_transform,
    #                                             parent_name="world",
    #                                             child_name="camera_position")
    #     # print(trans_from_0)
    #
    #     rate.sleep()
