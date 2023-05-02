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
from camera_calibration.utils.SaveMe import SaveMe
from camera_calibration.params.calibration import external_calibration_path_position
from camera_calibration.params.calibration import external_calibration_path
import camera_calibration.params.transform_frame_names as tfn

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

        self.listener = tf.TransformListener()
        # self.marker_subscriber = rospy.Subscriber('/detected_aruco_marker_ids', UInt8MultiArray, self.marker_callback)
        # self.tf_subscriber = rospy.Subscriber('/tf', TFMessage, self.callback)
        self.transforms_hand2world = []
        self.transforms_camera2aruco = []
        self.start_time = time()
        self.num_images_to_capture = 15

    def collect_transforms(self, num_images=None):
        if num_images is None:
            num_images = self.num_images_to_capture
        else:
            self.num_images_to_capture = num_images
        rate = rospy.Rate(1)
        # camera = "camera_to_aruco_[0]"

        while len(self.transforms_camera2aruco) < self.num_images_to_capture:
            # let the tfs start publishing
            rate.sleep()

            # Attached to gripper
            camera2aruco = self.get_transform_between(origin=tfn.camera, to=tfn.charuco)
            hand2world = self.get_transform_between(origin=tfn.hand, to=tfn.world)

            # Base to Camera
            # camera2aruco = self.get_transform_between(origin=aruco, to=camera)
            # hand2world = self.get_transform_between(origin=world, to=hand)

            input()
            print(camera2aruco)
            print(hand2world)
            if hand2world is not None and camera2aruco is not None:
                self.transforms_camera2aruco.append(camera2aruco)
                self.transforms_hand2world.append(hand2world)
            print(len(self.transforms_camera2aruco))

    @staticmethod
    def solve(fixed2attached, hand2base, solve_method, attached2hand_guess=None):
        # fixed = thing on table
        # attached = thing on arm
        # hand = gripper
        # bases = world
        # Solves AX=XB with hand2base being A and fixed2attached B

        # Fixed2Attached
        rot_fixed2attached, tran_fixed2attached = TypeConverter.transform_to_matrices(
            fixed2attached)

        # Hand2World
        rot_hand2world, tran_hand2world = TypeConverter.transform_to_matrices(
            hand2base)

        # Attached2Hand
        if attached2hand_guess is not None:
            # Init Guess Fixed2Hand
            rot_attached2hand_guess, trand_attached2hand_guess = TypeConverter.transform_to_matrices(
                attached2hand_guess
            )
            rot_attached2hand, tran_attached2hand = cv2.calibrateHandEye(
                R_gripper2base=rot_hand2world,
                t_gripper2base=tran_hand2world,
                R_target2cam=rot_fixed2attached,
                t_target2cam=tran_fixed2attached,
                R_cam2gripper=rot_attached2hand_guess,
                t_cam2gripper=trand_attached2hand_guess,
                method=solve_method
            )
        else:
            rot_attached2hand, tran_attached2hand = cv2.calibrateHandEye(
                R_gripper2base=rot_hand2world,
                t_gripper2base=tran_hand2world,
                R_target2cam=rot_fixed2attached,
                t_target2cam=tran_fixed2attached,
                method=solve_method
            )

        # print(rot_attached2hand, tran_attached2hand)
        return rot_attached2hand, tran_attached2hand

    def solve_all_sample_combos(self, solve_method=cv2.CALIB_HAND_EYE_TSAI):

        step_size = 1
        start_sample_size = 3
        end_sample_size = self.num_images_to_capture + 1

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
                    self.solve(
                        fixed2attached=camera2aruco_subset,
                        hand2base=hand2base_subset,
                        solve_method=solve_method
                    )
                )

        return poses

    def solve_all_algorithms(self):
        methods = [
            cv2.CALIB_HAND_EYE_TSAI,
            cv2.CALIB_HAND_EYE_PARK,
            cv2.CALIB_HAND_EYE_HORAUD,
            cv2.CALIB_HAND_EYE_ANDREFF,
            cv2.CALIB_HAND_EYE_DANIILIDIS
        ]
        poses = dict()

        for method in methods:
            poses[method] = list()
            poses[method].append(self.solve(
                fixed2attached=self.transforms_camera2aruco,
                hand2base=self.transforms_hand2world,
                solve_method=method)
            )

        return poses

    def get_transform_between(self, origin, to):
        try:
            transform = self.tfBuffer.lookup_transform(origin, to, rospy.Time())
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print(f"Oopsie! No transform between {origin} and {to} D:")
            return None

    @staticmethod
    def plot_pose_dict(pose_samples):

        sample_translations = dict()
        for sample_category, sample_poses in zip(pose_samples.keys(), pose_samples.values()):
            sample_translations[sample_category] = list()
            for rotation, translation in sample_poses:
                sample_translations[sample_category].append(translation)

        HarryPlotter.plot_translation_vector_categories(sample_translations)

    def save(self):
        SaveMe.save_transforms(self.transforms_camera2aruco, external_calibration_path + 'camera2aruco.json')
        SaveMe.save_transforms(self.transforms_hand2world, external_calibration_path + 'hand2world.json')

    def load(self):
        self.transforms_camera2aruco = SaveMe.load_transforms(external_calibration_path + 'camera2aruco.json')
        self.transforms_hand2world = SaveMe.load_transforms(external_calibration_path + 'hand2world.json')


if __name__ == '__main__':

    methods = [
        cv2.CALIB_HAND_EYE_TSAI,
        cv2.CALIB_HAND_EYE_PARK,
        cv2.CALIB_HAND_EYE_HORAUD,
        cv2.CALIB_HAND_EYE_ANDREFF,
        cv2.CALIB_HAND_EYE_DANIILIDIS
    ]

    save_data = True
    load_data = True
    rospy.init_node('hand_eye_node')
    hand_eye_estimator = EyeToHandEstimator()

    if load_data:
        save_data = False
        print('Calibrating camera position...')
        hand_eye_estimator.load()
    else:
        print('Press return to collect data point.')
        hand_eye_estimator.collect_transforms()
    if save_data:
        hand_eye_estimator.save()
        print('Saved data points.')

    # pose_estimations_samples = hand_eye_estimator.solve_all_sample_combos(solve_method=methods[0])
    pose_estimations_methods = hand_eye_estimator.solve_all_algorithms()

    rotation, translation = pose_estimations_methods[cv2.CALIB_HAND_EYE_TSAI][0]
    # print(rotation)
    rotation = TypeConverter.matrix_to_quaternion_vector(rotation)
    print('Calibration complete.')
    print(f'Camera was found at\nrotation:\n{rotation}\ntranslation:\n{translation}')

    transform = TypeConverter.vectors_to_stamped_transform(translation=translation, rotation=rotation,
                                                           parent_frame=tfn.world, child_frame=tfn.camera)
    transforms = [transform]
    SaveMe.save_transforms(transforms, external_calibration_path_position + 'camera_estimate.json')
    print(f'Saved transform frame to: {external_calibration_path_position}camera_estimate.json')

    # pub_tf_static = tf2_ros.StaticTransformBroadcaster()
    # TFPublish.publish_static_transform(publisher=pub_tf_static, parent_name=tfn.world, child_name=tfn.camera_estimate,
    #                                    rotation=rotation, translation=translation)

    # hand_eye_estimator.plot_pose_dict(pose_estimations_samples)
    # hand_eye_estimator.plot_pose_dict(pose_estimations_methods)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
