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
        self.transforms_hand2world = []
        self.transforms_camera2aruco = []
        self.start_time = time()

    def collect_transforms(self):
        rate = rospy.Rate(1)
        camera = "camera_to_aruco_[0]"
        aruco = "aruco_[0]"
        world = "world"
        hand = "panda_hand"
        while len(self.transforms_camera2aruco) < 5:
            # let the tfs start publishing
            rate.sleep()

            camera2aruco = self.get_transform_between(camera, aruco)
            hand2world = self.get_transform_between(hand, world)

            input()
            print(camera2aruco)
            print(hand2world)
            if hand2world is not None and camera2aruco is not None:
                self.transforms_camera2aruco.append(camera2aruco)
                self.transforms_hand2world.append(hand2world)
            print(len(self.transforms_camera2aruco))

    def solve(self):

        # fixed = camera, attached = aruco. in the example it's the other way around but should not matter?
        rot_fixed2attached, tran_fixed2attached = self.transform_to_matrices(self.transforms_camera2aruco)
        rot_hand2world, tran_hand2world = self.transform_to_matrices(self.transforms_hand2world)

        rot_attached2hand, tran_attached2hand = cv2.calibrateHandEye(
            R_gripper2base=rot_hand2world,
            t_gripper2base=tran_hand2world,
            R_target2cam=rot_fixed2attached,
            t_target2cam=tran_fixed2attached,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        print(rot_attached2hand, tran_attached2hand)

    @staticmethod
    def transform_to_matrices(transforms):
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

    def get_transform_between(self, source_frame, target_frame):
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


