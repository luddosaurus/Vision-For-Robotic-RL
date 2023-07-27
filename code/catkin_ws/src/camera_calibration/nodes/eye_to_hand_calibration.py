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
from camera_calibration.utils.ExtrinsicEvaluator import ExtrinsicEvaluator

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

        # live marker calibration mode
        self.toggle_marker_calibration = False
        self.marker_mode_memory = []
        self.live_camera_estimate_name = 'live_camera_estimate'
        self.live_estimate_result = None

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
        self.pub_live_estimate_position = tf2_ros.StaticTransformBroadcaster()
        self.pub_mean_live_estimate_position = tf2_ros.StaticTransformBroadcaster()
        # self.listener = tf.TransformListener()

        self.camera_subscriber = rospy.Subscriber(camera_topic, Image, self.camera_callback)

        self.cv_bridge = CvBridge()

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
        self.camera_estimates = []
        self.pose_estimations_all_algorithms = None

        self.memory_size = memory_size
        self.current_image = None
        self.Frame = Enum('Frame', 'camera charuco world panda_hand')
        self.eye_hand_solver = None

        self.eye_in_hand = eye_in_hand

        self.parent_frame_name = self.Frame.panda_hand.name if self.eye_in_hand else self.Frame.world.name

        if load_data_directory is not None:
            self.load(load_data_directory)
        self.save_directory = save_data_directory

    def camera_callback(self, input_image):
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(input_image, desired_encoding="bgr8")

        except CvBridgeError as e:
            print(e)

        self.collect_camera_target_transform()

        if self.toggle_marker_calibration:
            mean_translation, mean_rotation = MeanHelper.riemannian_mean(self.transform_memory)
            TFPublish.publish_static_stamped_transform(publisher=self.pub_live_estimate_position,
                                                       parent_name=self.Frame.charuco.name,
                                                       child_name=self.live_camera_estimate_name,
                                                       transform_stamped=TypeConverter.invert_transform_tf(
                                                           mean_translation, mean_rotation))

        # ---------------------- GUI
        info = "[q]uit " \
               "[s]ave " \
               "[u]ndo " \
               "[r]un_solver " \
               "[e]xtensive_run " \
               "[c]ollect " \
               "[t]oggle_mode"
        display_image = DaVinci.pad_image_cv(self.current_image)
        DaVinci.draw_text_box_in_corner(
            image=display_image,
            text=info,
            position="bottom_left",
            thickness=1,
            font_scale=0.8,
            background=(0, 0, 0)
        )
        DaVinci.draw_text_box_in_corner(
            image=display_image,
            text=f'Number of transforms captured: {len(self.transforms_camera2charuco)}',
            position='top_left',
            background=(0, 0, 0),
            thickness=1,
            font_scale=0.8,
        )

        cv2.imshow('External calibration display', display_image)

        # ---------------------- Input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit
            print('Shutting down....')
            rospy.signal_shutdown('We are done here')

        elif key == ord('c'):  # Collect
            if self.toggle_marker_calibration:
                self.marker_mode_memory.append(
                    self.get_transform_between(self.parent_frame_name, self.live_camera_estimate_name))
                print(f'collected value: {self.marker_mode_memory[-1]}')

            else:
                self.save_camera_target_transform()
                self.collect_robot_transforms()
                if len(self.transforms_camera2charuco) > len(self.transforms_hand2world):
                    self.transforms_camera2charuco = self.transforms_camera2charuco[:-1]
                    self.solve_all_methods()

        elif key == ord('u') and len(self.transforms_camera2charuco) > 0:  # Undo
            if self.toggle_marker_calibration:
                self.marker_mode_memory = self.marker_mode_memory[:-1]
            else:
                self.transforms_camera2charuco = self.transforms_camera2charuco[:-1]
                self.transforms_hand2world = self.transforms_hand2world[:-1]

        elif key == ord('e') and len(self.transforms_camera2charuco) >= 3:  # Run
            self.eye_hand_solver = EyeHandSolver(transforms_hand2world=self.transforms_hand2world,
                                                 transforms_camera2charuco=self.transforms_camera2charuco
                                                 )
            self.publish_all_methods()
            self.evaluate()

        elif key == ord('r') and (len(self.transforms_camera2charuco) >= 3 or len(self.marker_mode_memory) > 0):  # Plot

            if self.toggle_marker_calibration:
                mean_translation, mean_rotation = MeanHelper.riemannian_mean(self.marker_mode_memory)
                mean_live_estimate = TypeConverter.vectors_to_stamped_transform(translation=mean_translation,
                                                                                rotation=mean_rotation,
                                                                                parent_frame=self.parent_frame_name,
                                                                                child_frame='mean_live_camera_estimate',
                                                                                )

                self.live_estimate_result = mean_live_estimate
                print(self.live_estimate_result)

                TFPublish.publish_static_transform(publisher=self.pub_mean_live_estimate_position,
                                                   parent_name=self.parent_frame_name,
                                                   child_name='mean_live_camera_estimate',
                                                   translation=mean_translation,
                                                   rotation=mean_rotation)
            else:
                self.solve_all_methods()

                frame_methods = TypeConverter.convert_to_dataframe(self.pose_estimations_all_algorithms)
                self.calculate_mean_estimate()
                self.pretty_print_transforms(self.pose_estimations_all_algorithms)
                HarryPlotter.plot_poses(frame_methods)

        elif key == ord('s'):  # Save
            if self.toggle_marker_calibration:
                print('saving!')
                JSONHelper.save_live_estimate_result(eye_in_hand=self.eye_in_hand, estimate=self.live_estimate_result,
                                                     data_points=self.marker_mode_memory,
                                                     directory_name=self.save_directory)
            else:
                JSONHelper.save_extrinsic_data(eye_in_hand=self.eye_in_hand,
                                               camera2target=self.transforms_camera2charuco,
                                               hand2world=self.transforms_hand2world, estimates=self.camera_estimates,
                                               directory_name=self.save_directory)

        elif key == ord('t'):
            self.toggle_marker_calibration = not self.toggle_marker_calibration
        elif key == ord('p'):
            print(self.get_transform_between(self.Frame.world.name, self.live_camera_estimate_name))

    def solve_all_methods(self):
        self.eye_hand_solver = EyeHandSolver(transforms_hand2world=self.transforms_hand2world,
                                             transforms_camera2charuco=self.transforms_camera2charuco)
        if len(self.transforms_camera2charuco) >= 3:
            self.pose_estimations_all_algorithms = self.eye_hand_solver.solve_all_algorithms()

            self.camera_estimates = TypeConverter.estimates_to_transforms(self.pose_estimations_all_algorithms,
                                                                          self.parent_frame_name)

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

        average_stamped_transform = TypeConverter.vectors_to_stamped_transform(
            translation=mean_translation,
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
        if self.toggle_marker_calibration:
            self.marker_mode_memory = JSONHelper.load_live_estimate_data(load_data_directory, self.eye_in_hand)
        else:
            self.transforms_camera2charuco, self.transforms_hand2world = JSONHelper.load_extrinsic_data(
                load_data_directory,
                self.eye_in_hand)

    def publish_camera_estimates(self):
        rotation, translation = self.pose_estimations_all_algorithms[0][0]
        rotation = TypeConverter.matrix_to_quaternion_vector(rotation)
        pub_tf_static = tf2_ros.TransformBroadcaster()

        # pub_tf_static.sendTransform(self.camera_estimates[0])
        TFPublish.publish_static_transform(publisher=pub_tf_static,
                                           parent_name=self.parent_frame_name,
                                           child_name=f'camera_estimate0',
                                           rotation=rotation, translation=translation)

    def publish_each_method_estimate(self, method, rotation, translation):
        print(f'method: {method}\nrotation: {rotation}\ntranslation: {translation}')
        pub_tf_static = tf2_ros.StaticTransformBroadcaster()

        TFPublish.publish_static_transform(publisher=pub_tf_static, parent_name=self.parent_frame_name,
                                           child_name=f'camera_estimate{method}',
                                           rotation=rotation, translation=translation)

    def pretty_print_transforms(self, transforms):
        for method in self.methods:
            rotation, translation = transforms[method][0]
            rotation = TypeConverter.matrix_to_quaternion_vector(rotation)
            print(f'method: {method}\nrotation: {rotation}\ntranslation: {translation}')
            if not np.isnan(rotation).any() and not np.isnan(translation).any():
                pub_tf_static = tf2_ros.StaticTransformBroadcaster()

                for i in range(100):
                    TFPublish.publish_static_transform(publisher=pub_tf_static,
                                                       parent_name=self.parent_frame_name,
                                                       child_name=f'camera_estimate{method}',
                                                       rotation=rotation, translation=translation)

    def publish_all_methods(self):
        pose_estimations_methods = self.eye_hand_solver.solve_all_algorithms()

        # Find mean
        self.camera_estimates = TypeConverter.estimates_to_transforms(pose_estimations_methods, self.parent_frame_name)
        self.calculate_mean_estimate()

        # Publish each method estimation
        for method in self.methods:
            rotation, translation = pose_estimations_methods[method][0]
            rotation = TypeConverter.matrix_to_quaternion_vector(rotation)
            self.publish_each_method_estimate(method, rotation, translation)

    def evaluate(self):
        evaluator = ExtrinsicEvaluator(
            intrinsic_camera=self.camera_matrix,
            distortion=self.distortion,
            camera2charuco=self.transforms_camera2charuco,
            hand2world=self.transforms_hand2world
        )

        evaluator.evaluate2d(
            evaluation_type=ExtrinsicEvaluator.TYPE_ORDER,
            title="Image Count"
        )

        evaluator.evaluate2d(evaluation_type=ExtrinsicEvaluator.TYPE_RANDOM_AVG,
                             title="Image Count")

    def calculate_mean_estimate(self):
        mean_translation, mean_rotation = MeanHelper.riemannian_mean(self.camera_estimates)
        # print(mean_translation, mean_rotation)
        self.camera_estimates.append(TypeConverter.vectors_to_stamped_transform(translation=mean_translation,
                                                                                rotation=mean_rotation,
                                                                                parent_frame=self.parent_frame_name,
                                                                                child_frame='camera_estimate_MEAN'))


if __name__ == '__main__':

    config_file = rospy.get_param(param_name='hand_eye_node/config')

    board_name, camera_name, mode, camera_topic, memory_size, load_data_directory, save_data_directory = JSONHelper \
        .get_extrinsic_calibration_parameters(config_file)
    rospy.init_node('hand_eye_node')

    if mode == 'eye_in_hand':
        eye_in_hand = True
    else:
        eye_in_hand = False

    extrinsic_estimator = ExtrinsicEstimator(
        board_name=board_name, camera_name=camera_name,
        eye_in_hand=eye_in_hand,
        camera_topic=camera_topic,
        memory_size=memory_size, load_data_directory=load_data_directory,
        save_data_directory=save_data_directory)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')

    cv2.destroyAllWindows()
