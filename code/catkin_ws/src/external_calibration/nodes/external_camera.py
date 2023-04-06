#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

from external_calibration.utils.TFPublish import *
import geometry_msgs
import tf
import tf2_ros
from tf.transformations import quaternion_matrix
import numpy as np
from external_calibration.params.panda_hand_aruco import table_arucos, arm_arucos


# def get_numpy_from_transform(transform):
#     translation_array = np.array([transform.translation.x,
#                                   transform.translation.y,
#                                   transform.translation.z])
#     rotation_array = np.array([transform.rotaion.x,
#                                transform.rotaion.y,
#                                transform.rotaion.z,
#                                transform.rotaion.w])
#

def calculate_average_transform(transforms):
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


# Broadcast Transform: Base -> Camera
if __name__ == '__main__':
    rospy.init_node('external_camera_node')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
    # pub_aruco_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            transformations = list()
            aruco_list = arm_arucos + table_arucos
            for aruco in aruco_list:
                # create a tf listener
                tf_listener = tf.TransformListener()

                # define the source and target frames
                source_frame = 'world'
                target_frame = f'camera_from_{aruco.get_name()}'

                # define a timeout for the lookup operation
                timeout = rospy.Duration(1.0)
                # exists = True
                # try:
                #     # check if a transform exists from source_frame to target_frame
                #     exists, _, _ = tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), timeout)
                # except tf.Exception as ex:
                #     # the transform does not exist
                #     print(f"No transform exists from {source_frame} to {target_frame}: {ex}")
                #     continue
                # if exists:
                if tfBuffer.can_transform(target_frame, source_frame, rospy.Time(), timeout):
                    transform = tfBuffer.lookup_transform('world', f'camera_from_{aruco.get_name()}', rospy.Time())
                    print(transform)
                    transformations.append(transform)


            avg_transform = calculate_average_transform(transformations)
            # publish(trans, pub_aruco_tf)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        publish_static_transform_with_transform(publisher=pub_aruco_tf, transform_stamped=avg_transform,
                                                parent_name="world",
                                                child_name="camera_position")
        # print(trans_from_0)

        rate.sleep()
