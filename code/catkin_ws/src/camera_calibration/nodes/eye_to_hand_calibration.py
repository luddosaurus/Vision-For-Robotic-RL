#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy

import geometry_msgs
from tf2_msgs.msg import TFMessage
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import Image
import tf
import tf2_ros
import random
import pandas as pd

from enum import Enum

from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_matrix
import numpy as np
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from camera_calibration.utils.TypeConverter import TypeConverter
from camera_calibration.utils.HarryPlotterAndTheChamberOfSeaborn import HarryPlotter
from camera_calibration.utils.TFPublish import TFPublish
from camera_calibration.utils.SaveMe import SaveMe
from camera_calibration.utils.ErrorEstimator import ErrorEstimator
from camera_calibration.utils.ARHelper import ARHelper
from camera_calibration.utils.MeanHelper import MeanHelper
from camera_calibration.utils.DaVinci import DaVinci
from camera_calibration.utils.EyeHandSolver import EyeHandSolver

from camera_calibration.params.calibration import external_calibration_path_position
from camera_calibration.params.calibration import external_calibration_path
from camera_calibration.params.calibration import calibration_path_d455, calibration_path_d435
import camera_calibration.params.transform_frame_names as tfn

import cv2
from itertools import combinations


# GOAL find offset between ee and aruco
# subscribe to tf
# get camera-aruco, put in list A
# get base-ee, put in list B
# sync A and B so that A[i].time = B[i].time
# AX = XB . solve ee-aruco offset


# with np.load(calibration_path) as X:
#     intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]
#
# print("ArUcoFinder launched with internal parameters:")
# print(intrinsic_camera, distortion)


class EyeToHandEstimator(object):

    def __init__(self, charuco_board_shape, charuco_marker_size, charuco_square_size, dict_type, memory_size=30,
                 load_data=False, save_data=False, eye_in_hand=False):

        # self.intrinsic_camera, self.distortion =
        with np.load(calibration_path_d435 if eye_in_hand else calibration_path_d455) as X:
            self.intrinsic_camera, self.distortion, _, _ = [X[i] for i in
                                                            ('camMatrix', 'distCoef', 'rVector', 'tVector')]
        print("ArUcoFinder launched with internal parameters:")
        print(self.intrinsic_camera, self.distortion)

        self.transform_memory = []

        # self.average_r_vec = self.latest_r_vec
        # self.average_t_vec = self.latest_t_vec

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
        self.listener = tf.TransformListener()

        self.camera_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.camera_callback)
        # self.camera_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image,
        #                                           self.camera_callback)
        self.cv_bridge = CvBridge()
        self.arHelper = ARHelper(charuco_board_shape=charuco_board_shape, charuco_marker_size=charuco_marker_size,
                                 charuco_square_size=charuco_square_size, dict_type=dict_type)

        self.methods = [
            cv2.CALIB_HAND_EYE_TSAI,
            cv2.CALIB_HAND_EYE_PARK,
            cv2.CALIB_HAND_EYE_HORAUD,
            cv2.CALIB_HAND_EYE_ANDREFF,
            cv2.CALIB_HAND_EYE_DANIILIDIS
        ]

        # self.marker_subscriber = rospy.Subscriber('/detected_aruco_marker_ids', UInt8MultiArray, self.marker_callback)
        # self.tf_subscriber = rospy.Subscriber('/tf', TFMessage, self.callback)
        self.transforms_hand2world = []
        self.transforms_camera2charuco = []
        self.start_time = time()

        self.memory_size = memory_size
        self.current_image = None
        self.Frame = Enum('Frame', 'camera charuco world panda_hand')
        self.eye_hand_solver = None
        self.save_data = save_data
        self.eye_in_hand = eye_in_hand
        if load_data:
            self.load()

    def camera_callback(self, input_image):
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(input_image, desired_encoding="bgr8")

        except CvBridgeError as e:
            print(e)

        self.collect_camera_target_transform()

        DaVinci.draw_text(image=self.current_image,
                          text=f'Number of transforms captured: {len(self.transforms_camera2charuco)}')
        cv2.imshow('External calibration display', self.current_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print('Shutting down....')
            rospy.signal_shutdown('We are done here')
        elif key == ord('s'):
            self.save_camera_target_transform()
            self.collect_robot_transforms()

        elif key == ord('u') and len(self.transforms_camera2charuco) > 0:
            self.transforms_camera2charuco = self.transforms_camera2charuco[:-1]
            self.transforms_hand2world = self.transforms_hand2world[:-1]
        elif key == ord('d') and len(self.transforms_camera2charuco) > 3:
            self.eye_hand_solver = EyeHandSolver(transforms_hand2world=self.transforms_hand2world,
                                                 transforms_camera2charuco=self.transforms_camera2charuco,
                                                 number_of_transforms=len(self.transforms_camera2charuco))
            if self.save_data:
                self.save()
            self.run_solvers()

        elif key == ord('p') and len(self.transforms_camera2charuco) >= 3:

            self.eye_hand_solver = EyeHandSolver(transforms_hand2world=self.transforms_hand2world,
                                                 transforms_camera2charuco=self.transforms_camera2charuco,
                                                 nnumber_of_transforms=len(self.transforms_camera2charuco))
            pose_estimations_all_algorithms = self.eye_hand_solver.solve_all_algorithms()
            self.pretty_print_transforms(pose_estimations_all_algorithms)

        elif key == ord('h'):
            self.save()

    def collect_camera_target_transform(self):
        self.current_image, latest_r_vec, latest_t_vec = self.arHelper.estimate_charuco_pose(
            image=self.current_image,
            camera_matrix=self.intrinsic_camera,
            dist_coefficients=self.distortion)
        # print(latest_r_vec)
        # print(latest_t_vec)
        if self.eye_in_hand:
            # latest_t_vec, latest_r_vec = TypeConverter.invert_transform(latest_t_vec, latest_r_vec)
            latest_r_vec = TypeConverter.rotation_vector_to_quaternions(latest_r_vec)
        else:
            latest_r_vec = TypeConverter.rotation_vector_to_quaternions(latest_r_vec)
        latest_t_vec = np.array(latest_t_vec)
        latest_r_vec = np.array(latest_r_vec)
        if not np.isnan(latest_r_vec).any() and not np.isnan(latest_t_vec).any():
            transform = TypeConverter.vectors_to_stamped_transform(translation=latest_t_vec,
                                                                   rotation=latest_r_vec,
                                                                   parent_frame=self.Frame.camera.name,
                                                                   child_frame=self.Frame.charuco.name)
            self.transform_memory.append(transform)
            if len(self.transform_memory) > self.memory_size:
                self.transform_memory = self.transform_memory[1:]

    def save_camera_target_transform(self):

        mean_translation, mean_rotation = MeanHelper.riemannian_mean(self.transform_memory)
        # if self.eye_in_hand:
        #     mean_translation, mean_rotation = TypeConverter.invert_transform(mean_translation, mean_rotation)

        average_stamped_transform = TypeConverter.vectors_to_stamped_transform(translation=mean_translation,
                                                                               rotation=mean_rotation,
                                                                               parent_frame=self.Frame.camera.name,
                                                                               child_frame=self.Frame.charuco.name)

        # if self.eye_in_hand:
        #     average_stamped_transform = TypeConverter.invert_stamped_transform(average_stamped_transform)
        print(average_stamped_transform)
        self.transforms_camera2charuco.append(average_stamped_transform)

    def collect_robot_transforms(self):

        if self.eye_in_hand:
            origin = self.Frame.world.name
            child = self.Frame.panda_hand.name
        else:
            origin = self.Frame.panda_hand.name
            child = self.Frame.world.name

        hand2world = self.get_transform_between(origin=origin, to=child)
        if hand2world is not None:
            self.transforms_hand2world.append(hand2world)

    #
    #     rate = rospy.Rate(1)
    #     # camera = "camera_to_aruco_[0]"
    #     # camera = "charuco_to_camera"
    #     # aruco = "charuco"
    #     world = "world"
    #     hand = "panda_hand"
    #     while len(self.transforms_camera2aruco) < self.num_images_to_capture:
    #         # let the tfs start publishing
    #         rate.sleep()
    #         input()
    #         # Attached to gripper
    #         camera2aruco = self.get_transform_between(origin=camera, to=aruco)
    #         hand2world = self.get_transform_between(origin=hand, to=world)
    #
    #         # Base to Camera
    #         # camera2aruco = self.get_transform_between(origin=camera, to=aruco)
    #         # hand2world = self.get_transform_between(origin=world, to=hand)
    #
    #         # print(camera2aruco)
    #         # print(hand2world)
    #         if hand2world is not None and camera2aruco is not None:
    #             # self.transforms_camera2aruco.append(camera2aruco)
    #             self.transforms_hand2world.append(hand2world)
    #         # print(len(self.transforms_camera2aruco))
    #
    # # def run_solvers(self):
    #
    # @staticmethod
    # def solve(fixed2attached, hand2base, solve_method, attached2hand_guess=None):
    #     # fixed = thing on table
    #     # attached = thing on arm
    #     # hand = gripper
    #     # bases = world
    #     # Solves AX=XB with hand2base being A and fixed2attached B
    #
    #     # Fixed2Attached
    #     rot_fixed2attached, tran_fixed2attached = TypeConverter.transform_to_matrices(
    #         fixed2attached)
    #
    #     # Hand2World
    #     rot_hand2world, tran_hand2world = TypeConverter.transform_to_matrices(
    #         hand2base)
    #
    #     # Attached2Hand
    #     if attached2hand_guess is not None:
    #         # Init Guess Fixed2Hand
    #         rot_attached2hand_guess, trand_attached2hand_guess = TypeConverter.transform_to_matrices(
    #             attached2hand_guess
    #         )
    #         rot_attached2hand, tran_attached2hand = cv2.calibrateHandEye(
    #             R_gripper2base=rot_hand2world,
    #             t_gripper2base=tran_hand2world,
    #             R_target2cam=rot_fixed2attached,
    #             t_target2cam=tran_fixed2attached,
    #             R_cam2gripper=rot_attached2hand_guess,
    #             t_cam2gripper=trand_attached2hand_guess,
    #             method=solve_method
    #         )
    #     else:
    #         try:
    #             rot_attached2hand, tran_attached2hand = cv2.calibrateHandEye(
    #                 R_gripper2base=rot_hand2world,
    #                 t_gripper2base=tran_hand2world,
    #                 R_target2cam=rot_fixed2attached,
    #                 t_target2cam=tran_fixed2attached,
    #                 method=solve_method
    #             )
    #         except:
    #             print('bad value')
    #             return None, None
    #
    #     # print(rot_attached2hand, tran_attached2hand)
    #     return rot_attached2hand, tran_attached2hand
    #
    # def solve_all_sample_combos(
    #         self,
    #         solve_method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    #         start_sample_size=15,
    #         end_sample_size=21,
    #         step_size=1):
    #
    #     if end_sample_size is None:
    #         end_sample_size = self.num_images_to_capture + 1
    #
    #     poses = dict()
    #     list_size = len(self.transforms_camera2aruco)
    #     max_iterations = 0
    #     # For every sample size
    #     for sample_size in range(start_sample_size, end_sample_size, step_size):
    #         print(sample_size)
    #         poses[sample_size] = list()
    #
    #         # For every index combination
    #         for sample_indices in combinations(range(list_size), sample_size):
    #             # Take out subset of indices
    #             camera2aruco_subset = [self.transforms_camera2aruco[index] for index in sample_indices]
    #             hand2base_subset = [self.transforms_hand2world[index] for index in sample_indices]
    #
    #             # Do and save estimation
    #             rot, tran = self.solve(
    #                 fixed2attached=camera2aruco_subset,
    #                 hand2base=hand2base_subset,
    #                 solve_method=solve_method
    #             )
    #             if rot is not None and tran is not None:
    #                 poses[sample_size].append(
    #                     (rot, tran)
    #                 )
    #             max_iterations += 1
    #             if max_iterations >= 300:
    #                 break
    #         max_iterations = 0
    #
    #     return poses
    #
    # def solve_all_method_samples(
    #         self,
    #         solve_methods,
    #         start_sample_size=20,
    #         end_sample_size=None,
    #         step_size=1):
    #
    #     # Solve all sample sizes for each algorithm
    #     if end_sample_size is None:
    #         end_sample_size = self.num_images_to_capture + 1
    #     poses = dict()
    #     max_iterations = 0
    #     for method in solve_methods:
    #         poses[method] = list()
    #
    #         for sample_size in range(start_sample_size, end_sample_size, step_size):
    #             sample_indices = random.sample(range(len(self.transforms_camera2aruco)), sample_size)
    #             camera2aruco_subset = [self.transforms_camera2aruco[index] for index in sample_indices]
    #             hand2base_subset = [self.transforms_hand2world[index] for index in sample_indices]
    #
    #             poses[method].append(
    #                 self.solve(
    #                     fixed2attached=camera2aruco_subset,
    #                     hand2base=hand2base_subset,
    #                     solve_method=method
    #                 )
    #             )
    #             max_iterations += 1
    #             if max_iterations >= 300:
    #                 break
    #         max_iterations = 0
    #
    #     return poses
    #
    # def solve_all_algorithms(self, solve_methods):
    #
    #     poses = dict()
    #
    #     for method in solve_methods:
    #         poses[method] = list()
    #         poses[method].append(
    #             self.solve(
    #                 fixed2attached=self.transforms_camera2aruco,
    #                 hand2base=self.transforms_hand2world,
    #                 solve_method=method
    #             )
    #         )
    #
    #     return poses

    def get_transform_between(self, origin, to):
        try:
            transform = self.tfBuffer.lookup_transform(origin, to, rospy.Time())
            print(transform)
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print(f"Oopsie! No transform between {origin} and {to} D:")
            return None

    # @staticmethod
    # def plot_pose_dict(pose_samples):
    #
    #     sample_translations = dict()
    #     for sample_category, poses in zip(pose_samples.keys(), pose_samples.values()):
    #         sample_translations[sample_category] = list()
    #
    #         for _, t_vec in poses:
    #             sample_translations[sample_category].append(t_vec)
    #
    #     # HarryPlotter.plot_translation_vector_categories(sample_translations)
    #     HarryPlotter.plot_translation_vectors_gradient(sample_translations)

    def save(self):
        SaveMe.save_transforms(self.transforms_camera2charuco, external_calibration_path + 'camera2charuco.json')
        SaveMe.save_transforms(self.transforms_hand2world, external_calibration_path + 'hand2world.json')

    def load(self):
        self.transforms_camera2charuco = SaveMe.load_transforms(external_calibration_path + 'camera2charuco.json')
        self.transforms_hand2world = SaveMe.load_transforms(external_calibration_path + 'hand2world.json')

    def pretty_print_transforms(self, transforms):
        print(transforms)
        for method in self.methods:
            # if method == self.methods[3]:
            #     continue
            rotation, translation = transforms[method][0]
            rotation = TypeConverter.matrix_to_quaternion_vector(rotation)
            print(f'method: {method}\nrotation: {rotation}\ntranslation: {translation}')
            if not np.isnan(rotation).any() and not np.isnan(translation).any():
                pub_tf_static = tf2_ros.StaticTransformBroadcaster()
                parent_frame = self.Frame.panda_hand.name if self.eye_in_hand else self.Frame.world.name
                TFPublish.publish_static_transform(publisher=pub_tf_static,
                                                   parent_name=parent_frame,
                                                   child_name=f'camera_estimate{method}',
                                                   rotation=rotation, translation=translation)

    def run_solvers(self):
        pose_estimations_samples = self.eye_hand_solver.solve_all_sample_combos(solve_method=self.methods[0])
        pose_estimations_methods = self.eye_hand_solver.solve_all_algorithms()
        pose_estimations_method_samples = self.eye_hand_solver.solve_all_method_samples()

        for method in self.methods:
            # if method == self.methods[3]:
            #     continue
            rotation, translation = pose_estimations_methods[method][0]
            rotation = TypeConverter.matrix_to_quaternion_vector(rotation)
            # ---------------------------- Test mean of all methods

            # ---------------------------- Publish
            # print(rotation)

            # print('Calibration complete.')
            # print(f'Camera was found at\nrotation:\n{rotation}\ntranslation:\n{translation}')
            #

            print(f'method: {method}\nrotation: {rotation}\ntranslation: {translation}')
            pub_tf_static = tf2_ros.StaticTransformBroadcaster()
            TFPublish.publish_static_transform(publisher=pub_tf_static, parent_name="world",
                                               child_name=f'camera_estimate{method}',
                                               rotation=rotation, translation=translation)

        # ---------------------------- Plot

        # ---------------------------- Convert to pandas
        # Frame [Category, Translation XYZ, Rotation XYZW]
        frame_samples = TypeConverter.convert_to_dataframe(pose_estimations_samples)
        frame_methods = TypeConverter.convert_to_dataframe(pose_estimations_methods)
        frame_method_samples = TypeConverter.convert_to_dataframe(pose_estimations_method_samples)

        # ---------------------------- Plot 3D
        HarryPlotter.plot_3d_scatter(frame_samples)
        HarryPlotter.plot_3d_scatter(frame_methods)
        HarryPlotter.plot_3d_scatter(frame_method_samples)

        # ---------------------------- Plot 2D
        true_translation = translation
        true_rotation = rotation

        # Standard Deviations
        frame_std = ErrorEstimator.calculate_standard_deviation_by_category(frame_samples)

        # Distance
        frame_distance = ErrorEstimator.calculate_distance_to_truth(frame_samples, true_translation)
        HarryPlotter.plot_histogram_by_category(frame_distance)


if __name__ == '__main__':

    Board = Enum('Board', 'small medium large')

    rospy.init_node('hand_eye_node')

    selected_board = Board.small

    save = False
    load = True

    if selected_board == Board.small:
        hand_eye_estimator = EyeToHandEstimator(charuco_board_shape=(7, 10), charuco_square_size=0.012,
                                                charuco_marker_size=0.008,
                                                dict_type=cv2.aruco.DICT_4X4_50, load_data=load, save_data=save,
                                                eye_in_hand=False)
    elif selected_board == Board.medium:
        hand_eye_estimator = EyeToHandEstimator(charuco_board_shape=(18, 29), charuco_square_size=0.01,
                                                charuco_marker_size=0.008,
                                                dict_type=cv2.aruco.DICT_5X5_1000, load_data=load, save_data=save)
    elif selected_board == Board.large:
        hand_eye_estimator = EyeToHandEstimator(charuco_board_shape=(9, 14), charuco_square_size=0.04,
                                                charuco_marker_size=0.031,
                                                dict_type=cv2.aruco.DICT_5X5_100, load_data=load, save_data=save,
                                                eye_in_hand=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()
