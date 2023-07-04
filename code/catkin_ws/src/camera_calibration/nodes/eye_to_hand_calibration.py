#! /usr/bin/env python3.8

import rospy

from sensor_msgs.msg import Image
import tf
import tf2_ros

from enum import Enum

from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from time import time

from camera_calibration.utils.TypeConverter import TypeConverter
from camera_calibration.utils.HarryPlotterAndTheChamberOfSeaborn import HarryPlotter
from camera_calibration.utils.TFPublish import TFPublish
from camera_calibration.utils.ErrorEstimator import ErrorEstimator
from camera_calibration.utils.ARHelper import ARHelper
from camera_calibration.utils.MeanHelper import MeanHelper
from camera_calibration.utils.DaVinci import DaVinci
from camera_calibration.utils.EyeHandSolver import EyeHandSolver
from camera_calibration.params.aruco_dicts import ARUCO_DICT
from camera_calibration.utils.JSONHelper import JSONHelper

import cv2


class ExtrinsicEstimator(object):

    def __init__(self, board_name, camera_name, camera_topic, eye_in_hand, memory_size, load_data_directory,
                 save_data_directory):

        camera_intrinsics = JSONHelper.get_camera_intrinsics(camera_name)
        board_data = JSONHelper.get_board_parameters(board_name)

        self.camera_matrix = np.array(camera_intrinsics['camera_matrix'])
        self.distortion = np.array(camera_intrinsics['distortion'])
        print(self.camera_matrix, self.distortion)

        self.transform_memory = []

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
        self.pub_charuco_position = tf2_ros.StaticTransformBroadcaster()
        # self.listener = tf.TransformListener()

        self.camera_subscriber = rospy.Subscriber(camera_topic, Image, self.camera_callback)

        self.cv_bridge = CvBridge()
        print(board_data)
        self.arHelper = ARHelper(charuco_board_shape=board_data['board_shape'],
                                 charuco_marker_size=board_data['aruco_marker_size'],
                                 charuco_square_size=board_data['square_size'],
                                 dict_type=cv2.aruco.getPredefinedDictionary(ARUCO_DICT[board_data['dict_type']]))

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
        self.camera_estimates = {}
        self.pose_estimations_all_algorithms = None

        self.start_time = time()

        self.memory_size = memory_size
        self.current_image = None
        self.Frame = Enum('Frame', 'camera charuco world panda_hand')
        self.eye_hand_solver = None

        self.eye_in_hand = eye_in_hand
        self.cameras_published = False
        self.published = False

        if load_data_directory is not None:
            self.load(load_data_directory)
        self.save_directory = save_data_directory

    def camera_callback(self, input_image):
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(input_image, desired_encoding="bgr8")

        except CvBridgeError as e:
            print(e)

        self.collect_camera_target_transform()

        if self.cameras_published:
            # self.publish_camera_estimates()
            TFPublish.publish_static_stamped_transform(publisher=self.pub_charuco_position,
                                                       parent_name='camera_estimate0',
                                                       child_name='charuco',
                                                       transform_stamped=self.transform_memory[-1])

        # ---------------------- GUI
        info = "[q]uit " \
               "[s]ave " \
               "[u]ndo " \
               "[r]un " \
               "[p]lot " \
               "[c]ollect"
        DaVinci.draw_text_box_in_corner(
            image=self.current_image,
            text=info,
            position="bottom_left",
            thickness=1,
            font_scale=0.8
        )
        DaVinci.draw_text_box_in_corner(
            image=self.current_image,
            text=f'Number of transforms captured: {len(self.transforms_camera2charuco)}',
            position='top_left'
        )
        # resized_image = DaVinci.resize(self.current_image.copy())
        cv2.imshow('External calibration display', self.current_image)

        # ---------------------- Input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit
            print('Shutting down....')
            rospy.signal_shutdown('We are done here')

        elif key == ord('c'):  # Collect
            self.save_camera_target_transform()
            self.collect_robot_transforms()
            if len(self.transforms_camera2charuco) > len(self.transforms_hand2world):
                self.transforms_camera2charuco = self.transforms_camera2charuco[:-1]
                self.solve_all_methods()


        elif key == ord('u') and len(self.transforms_camera2charuco) > 0:  # Undo
            self.transforms_camera2charuco = self.transforms_camera2charuco[:-1]
            self.transforms_hand2world = self.transforms_hand2world[:-1]

        elif key == ord('r') and len(self.transforms_camera2charuco) >= 3:  # Run
            self.eye_hand_solver = EyeHandSolver(transforms_hand2world=self.transforms_hand2world,
                                                 transforms_camera2charuco=self.transforms_camera2charuco,
                                                 number_of_transforms=len(self.transforms_camera2charuco))
            self.run_solvers()

        elif key == ord('p') and len(self.transforms_camera2charuco) >= 3:  # Plot
            self.solve_all_methods()
            self.camera_estimates = TypeConverter.estimates_to_transforms(self.pose_estimations_all_algorithms)
            self.pretty_print_transforms(self.pose_estimations_all_algorithms)
            frame_methods = TypeConverter.convert_to_dataframe(self.pose_estimations_all_algorithms)
            self.calculate_mean_estimate()
            HarryPlotter.plot_poses(frame_methods)
            self.cameras_published = True

        elif key == ord('s'):  # Save
            JSONHelper.save_extrinsic_data(eye_in_hand=self.eye_in_hand, camera2target=self.transforms_camera2charuco,
                                           hand2world=self.transforms_hand2world, estimates=self.camera_estimates,
                                           directory_name=self.save_directory)
            # self.save()

    def solve_all_methods(self):
        self.eye_hand_solver = EyeHandSolver(transforms_hand2world=self.transforms_hand2world,
                                             transforms_camera2charuco=self.transforms_camera2charuco,
                                             number_of_transforms=len(self.transforms_camera2charuco))
        if len(self.transforms_camera2charuco) >= 3:
            self.pose_estimations_all_algorithms = self.eye_hand_solver.solve_all_algorithms()
            self.camera_estimates = TypeConverter.estimates_to_transforms(self.pose_estimations_all_algorithms)
            self.pretty_print_transforms(self.pose_estimations_all_algorithms)

    def collect_camera_target_transform(self):
        self.current_image, latest_r_vec, latest_t_vec = self.arHelper.estimate_charuco_pose(
            image=self.current_image,
            camera_matrix=self.camera_matrix,
            dist_coefficients=self.distortion)

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

        average_stamped_transform = TypeConverter.vectors_to_stamped_transform(translation=mean_translation,
                                                                               rotation=mean_rotation,
                                                                               parent_frame=self.Frame.camera.name,
                                                                               child_frame=self.Frame.charuco.name)

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

    def get_transform_between(self, origin, to):
        try:
            transform = self.tfBuffer.lookup_transform(origin, to, rospy.Time())

            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print(f"No transform found between {origin} and {to}.")
            return None

    def load(self, load_data_directory):
        self.transforms_camera2charuco, self.transforms_hand2world = JSONHelper.load_extrinsic_data(load_data_directory,
                                                                                                    self.eye_in_hand)

    def publish_camera_estimates(self):
        rotation, translation = self.pose_estimations_all_algorithms[0][0]
        rotation = TypeConverter.matrix_to_quaternion_vector(rotation)
        pub_tf_static = tf2_ros.TransformBroadcaster()
        parent_frame = self.Frame.panda_hand.name if self.eye_in_hand else self.Frame.world.name
        # pub_tf_static.sendTransform(self.camera_estimates[0])
        TFPublish.publish_static_transform(publisher=pub_tf_static,
                                           parent_name=parent_frame,
                                           child_name=f'camera_estimate0',
                                           rotation=rotation, translation=translation)

    def pretty_print_transforms(self, transforms):
        for method in self.methods:
            rotation, translation = transforms[method][0]
            rotation = TypeConverter.matrix_to_quaternion_vector(rotation)
            print(f'method: {method}\nrotation: {rotation}\ntranslation: {translation}')
            if not np.isnan(rotation).any() and not np.isnan(translation).any():
                pub_tf_static = tf2_ros.StaticTransformBroadcaster()
                parent_frame = self.Frame.panda_hand.name if self.eye_in_hand else self.Frame.world.name
                for i in range(100):
                    TFPublish.publish_static_transform(publisher=pub_tf_static,
                                                       parent_name=parent_frame,
                                                       child_name=f'camera_estimate{method}',
                                                       rotation=rotation, translation=translation)
                self.published = True

    def run_solvers(self):
        pose_estimations_samples = self.eye_hand_solver.solve_all_sample_combos(solve_method=self.methods[0])
        pose_estimations_methods = self.eye_hand_solver.solve_all_algorithms()
        pose_estimations_method_samples = self.eye_hand_solver.solve_all_method_samples()
        self.camera_estimates = TypeConverter.estimates_to_transforms(pose_estimations_methods)
        self.calculate_mean_estimate()
        for method in self.methods:
            rotation, translation = pose_estimations_methods[method][0]
            rotation = TypeConverter.matrix_to_quaternion_vector(rotation)
            # ---------------------------- Test mean of all methods

            # ---------------------------- Publish each methods estimate

            print(f'method: {method}\nrotation: {rotation}\ntranslation: {translation}')
            pub_tf_static = tf2_ros.StaticTransformBroadcaster()
            parent_frame = self.Frame.panda_hand.name if self.eye_in_hand else self.Frame.world.name
            TFPublish.publish_static_transform(publisher=pub_tf_static, parent_name=parent_frame,
                                               child_name=f'camera_estimate{method}',
                                               rotation=rotation, translation=translation)

        # ---------------------------- Plot

        # ---------------------------- Convert to pandas
        # Frame [Category, Translation XYZ, Rotation XYZW]
        frame_samples = TypeConverter.convert_to_dataframe(pose_estimations_samples)
        frame_methods = TypeConverter.convert_to_dataframe(pose_estimations_methods)
        frame_method_samples = TypeConverter.convert_to_dataframe(pose_estimations_method_samples)

        # --------------------------- Plot frame poses
        HarryPlotter.plot_poses(frame_methods)

        # ---------------------------- Plot 3D
        # HarryPlotter.plot_3d_scatter(frame_samples)
        # HarryPlotter.plot_3d_scatter(frame_methods)
        # HarryPlotter.plot_3d_scatter(frame_method_samples)

        # ---------------------------- Plot 2D
        true_translation = np.array(translation).flatten()
        true_rotation = rotation

        # Standard Deviations
        frame_std = ErrorEstimator.calculate_standard_deviation_by_category(frame_samples)

        # Distance density (Truth to Pose estimations)
        frame_distance = ErrorEstimator.calculate_distance_to_truth(frame_samples, true_translation)
        HarryPlotter.plot_histogram_by_category(frame_distance)
        HarryPlotter.plot_prop(frame_distance, x='Distance')

        # Each value
        HarryPlotter.plot_prop(frame_samples, x='Translation X')
        HarryPlotter.plot_prop(frame_samples, x='Translation Y')
        HarryPlotter.plot_prop(frame_samples, x='Translation Z')

        # Distance density (World Center to Pose estimations)
        translation_columns = ["Translation X", "Translation Y", "Translation Z"]
        HarryPlotter.plot_distance_density(frame_samples, translation_columns)

        rotation_columns = ["Rotation X", "Rotation Y", "Rotation Z", "Rotation W"]
        HarryPlotter.plot_distance_density(frame_samples, rotation_columns)

        # # Variance
        # frame_variance = ErrorEstimator.calculate_variance_by_category(frame_samples)
        #
        # HarryPlotter.stacked_histogram(frame_variance)

    def calculate_mean_estimate(self):
        mean_translation, mean_rotation = MeanHelper.riemannian_mean(self.camera_estimates)
        print(mean_translation, mean_rotation)
        self.camera_estimates.append(TypeConverter.vectors_to_stamped_transform(translation=mean_translation,
                                                                                rotation=mean_rotation,
                                                                                parent_frame=self.Frame.world.name,
                                                                                child_frame='camera_estimate_MEAN'))


if __name__ == '__main__':

    config_file = rospy.get_param(param_name='hand_eye_node/config')

    board_name, camera_name, mode, camera_topic, memory_size, load_data_directory, save_data_directory = JSONHelper.get_extrinsic_calibration_parameters(
        config_file)
    rospy.init_node('hand_eye_node')
    if mode == 'eye_in_hand':
        eye_in_hand = True
    else:
        eye_in_hand = False
    print(eye_in_hand)
    extrinsic_estimator = ExtrinsicEstimator(board_name=board_name, camera_name=camera_name,
                                             eye_in_hand=eye_in_hand,
                                             camera_topic=camera_topic,
                                             memory_size=memory_size, load_data_directory=load_data_directory,
                                             save_data_directory=save_data_directory)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()
