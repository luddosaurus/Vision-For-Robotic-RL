#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy
from camera_calibration.utils.TFPublish import *
import geometry_msgs
from tf2_msgs.msg import TFMessage
from std_msgs.msg import UInt8MultiArray
import tf
import tf2_ros

from tf.transformations import quaternion_matrix
import numpy as np
from time import time
from camera_calibration.utils.TFTransformer import TFTransformer
from camera_calibration.utils.HandEyeCalibrator import HandEyeCalibrator
import cv2


# GOAL find offset between ee and aruco
# subscribe to tf
# get camera-aruco, put in list A
# get base-ee, put in list B
# sync A and B so that A[i].time = B[i].time
# AX = XB . solve ee-aruco offset
class EyeToHandEstimator(object):
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
        self.listener = tf.TransformListener()
        # self.marker_subscriber = rospy.Subscriber('/detected_aruco_marker_ids', UInt8MultiArray, self.marker_callback)
        # self.tf_subscriber = rospy.Subscriber('/tf', TFMessage, self.callback)
        self.world_to_panda_hand_transforms = []
        self.camera_to_aruco_transforms = []
        self.start_time = time()

        # todo
        # Get world-ee frames
        # Get camera-aruco frames done!
        # Sync frames
        # AX = XB ish

    def collect_transforms(self):
        rate = rospy.Rate(1)
        camera = "camera_to_aruco_[0]"
        aruco = "aruco_[0]"
        world = "world"
        hand = "panda_hand"
        while len(self.camera_to_aruco_transforms) < 5:
            # let the tfs start publishing
            rate.sleep()

            transform_camera_to_aruco = self.get_transform(camera, aruco)

            transform_world_to_panda_hand = self.get_transform(world, hand)
            input()
            print(transform_camera_to_aruco)
            print(transform_world_to_panda_hand)
            if transform_world_to_panda_hand is not None and transform_camera_to_aruco is not None:
                self.camera_to_aruco_transforms.append(transform_camera_to_aruco)
                self.world_to_panda_hand_transforms.append(transform_camera_to_aruco)
            print(len(self.camera_to_aruco_transforms))

    def solve(self):
        camera_matrices = TFTransformer.stamped_transforms_to_matrices(self.camera_to_aruco_transforms)
        panda_hand_matrices = TFTransformer.stamped_transforms_to_matrices(self.world_to_panda_hand_transforms)

        rot_cam, tran_cam = self.transform_to_matrices(self.camera_to_aruco_transforms)
        rot_hand, tran_hand = self.transform_to_matrices(self.world_to_panda_hand_transforms)

        rot, tran = cv2.calibrateHandEye(rot_hand, tran_hand, rot_cam, tran_cam)
        print(rot, tran)
        # something = HandEyeCalibrator.hand_eye_calibration(panda_hand_matrices, camera_matrices)
        # print(something)

    def transform_to_matrices(self, transforms):
        rotation_matrices = []
        translations = []
        for transform in transforms:
            # Get the translation and quaternion from the transform message
            translation = np.array([transform.transform.translation.x,
                                    transform.transform.translation.y,
                                    transform.transform.translation.z])
            quaternion = np.array([transform.transform.rotation.x,
                                   transform.transform.rotation.y,
                                   transform.transform.rotation.z,
                                   transform.transform.rotation.w])
            # Convert the quaternion to a rotation matrix
            rotation_matrix = quaternion_matrix(quaternion)[:3, :3]
            # Append the rotation matrix and translation to the output lists
            rotation_matrices.append(rotation_matrix)
            translations.append(translation)
        return rotation_matrices, translations

    def get_transform(self, source_frame, target_frame):
        try:
            transform = self.tfBuffer.lookup_transform(source_frame, target_frame, rospy.Time())
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None


if __name__ == '__main__':
    rospy.init_node('hand_eye_node')
    hand_eye_estimator = EyeToHandEstimator()
    hand_eye_estimator.collect_transforms()
    hand_eye_estimator.solve()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')


def randomness():
    while not rospy.is_shutdown():
        try:
            (base_to_ee_translation, base_to_ee_rotation) = listener.lookupTransform(
                'base_frame', 'end_effector_frame', rospy.Time(0))

            # (camera_to_aruco_translation, camera_to_aruco_rotation) = listener.lookupTransform(
            #     'camera_frame', 'aruco_[0]', rospy.Time(0))

            base_to_ee_transforms.append((base_to_ee_translation, base_to_ee_rotation))
            camera_to_aruco_transforms.append((camera_to_aruco_translation, camera_to_aruco_rotation))

            rospy.sleep(0.01)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

    synchronized_transforms = []

    for base_to_ee_transform in base_to_ee_transforms:
        # Find the closest camera to aruco transform by time stamp
        closest_camera_to_aruco_transform = min(
            camera_to_aruco_transforms,
            key=lambda x: abs(x[0].to_sec() - base_to_ee_transform[0].to_sec()))

        # Append the synchronized transform
        synchronized_transforms.append((
            base_to_ee_transform[0],
            base_to_ee_transform[1],
            closest_camera_to_aruco_transform[0],
            closest_camera_to_aruco_transform[1]))
