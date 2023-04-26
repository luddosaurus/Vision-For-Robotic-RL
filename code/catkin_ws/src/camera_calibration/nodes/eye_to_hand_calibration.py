#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy

import geometry_msgs
from tf2_msgs.msg import TFMessage
from std_msgs.msg import UInt8MultiArray
import tf
import tf2_ros

from tf.transformations import quaternion_matrix
import numpy as np
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from camera_calibration.utils.TypeConverter import TypeConverter
from camera_calibration.utils.HarryPlotter import HarryPlotter
from camera_calibration.utils.TFPublish import TFPublish

import cv2
from itertools import combinations


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
        self.num_images_to_capture = 5

    def collect_transforms(self):
        rate = rospy.Rate(1)
        # camera = "camera_to_aruco_[0]"
        camera = "charuco_to_camera"
        aruco = "charuco"
        world = "world"
        hand = "panda_hand"
        while len(self.transforms_camera2aruco) < self.num_images_to_capture:
            # let the tfs start publishing
            rate.sleep()

            # Attached to gripper
            # camera2aruco = self.get_transform_between(origin=camera, to=aruco)
            # hand2world = self.get_transform_between(origin=hand, to=world)

            # Base to Camera
            camera2aruco = self.get_transform_between(origin=aruco, to=camera)
            hand2world = self.get_transform_between(origin=world, to=hand)

            input()
            print(camera2aruco)
            print(hand2world)
            if hand2world is not None and camera2aruco is not None:
                self.transforms_camera2aruco.append(camera2aruco)
                self.transforms_hand2world.append(hand2world)
            print(len(self.transforms_camera2aruco))

    def solve_sample(self, fixed2attached, hand2base):

        # Fixed2Attached
        rot_fixed2attached, tran_fixed2attached = TypeConverter.transform_to_matrices(
            fixed2attached)

        # Hand2World
        rot_hand2world, tran_hand2world = TypeConverter.transform_to_matrices(
            hand2base)

        # Attached2Hand
        rot_attached2hand, tran_attached2hand = cv2.calibrateHandEye(
            R_gripper2base=rot_hand2world,
            t_gripper2base=tran_hand2world,
            R_target2cam=rot_fixed2attached,
            t_target2cam=tran_fixed2attached,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        print(rot_attached2hand, tran_attached2hand)
        return rot_attached2hand, tran_attached2hand

    def solve(self):

        step_size = 1
        start_sample_size = 3
        end_sample_size = self.num_images_to_capture

        poses = dict()
        list_size = len(self.transforms_camera2aruco)

        # For every sample size
        for sample_size in range(start_sample_size, end_sample_size, step_size):
            poses[sample_size] = list()

            # For every index combination
            for sample_indices in combinations(range(list_size), sample_size):
                # Take out subset of indices
                camera2aruco_subset = [self.transforms_camera2aruco[index] for index in sample_indices]
                hand2base_subset = [self.transforms_hand2world[index] for index in sample_indices]

                # Do and save estimation
                poses[sample_size].append(
                    self.solve_sample(
                        fixed2attached=camera2aruco_subset,
                        hand2base=hand2base_subset
                    )
                )

        return poses

    def get_transform_between(self, origin, to):
        try:
            transform = self.tfBuffer.lookup_transform(origin, to, rospy.Time())
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print(f"Oopsie! No transform between {origin} and {to} D:")
            return None

    def plot_translation_vectors(self, translations):
        # Create a new figure and axis for 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract x, y, and z values from each translation vector
        x_vals = [t[0] for t in translations]
        y_vals = [t[1] for t in translations]
        z_vals = [t[2] for t in translations]

        # Plot the translation vectors as a scatter cloud
        ax.scatter(x_vals, y_vals, z_vals)

        # Set the axis labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        # Show the plot
        plt.show()


if __name__ == '__main__':
    rospy.init_node('hand_eye_node')
    hand_eye_estimator = EyeToHandEstimator()
    hand_eye_estimator.collect_transforms()
    pose_estimations = hand_eye_estimator.solve()

    translations = list()

    for sample in pose_estimations.values():

        for t in sample:
            translations.append(t[1])

    hand_eye_estimator.plot_translation_vectors(translations)
    #   HarryPlotter.plot_distances(
    #    distance_dict=pose_estimations,
    #    use_box=False
    # )

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
