#! /usr/bin/env python3.8
import os

import rospkg
import rospy
from sensor_msgs.msg import Image
import tf
import tf2_ros

import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from camera_calibration.utils.JSONHelper import JSONHelper
from camera_calibration.utils.TypeConverter import TypeConverter
from camera_calibration.utils.ARHelper import ARHelper
from camera_calibration.params.aruco_dicts import ARUCO_DICT
from camera_calibration.utils.TFPublish import TFPublish


class CharucoPublisher(object):
    def __init__(self, board_name, camera_name, camera_topic):

        self.aruco_mode = True
        camera_intrinsics = JSONHelper.get_camera_intrinsics(camera_name)
        board_data = JSONHelper.get_board_parameters(board_name)

        self.camera_matrix = np.array(camera_intrinsics['camera_matrix'])
        self.distortion = np.array(camera_intrinsics['distortion'])
        print(self.camera_matrix, self.distortion)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
        self.pub_charuco_position = tf2_ros.StaticTransformBroadcaster()

        self.camera_subscriber = rospy.Subscriber(camera_topic, Image, self.camera_callback)

        self.cv_bridge = CvBridge()
        self.current_image = None

        self.transform_memory = []
        self.memory_size = 20

        self.arHelper = ARHelper(charuco_board_shape=board_data['board_shape'],
                                 charuco_marker_size=board_data['aruco_marker_size'],
                                 charuco_square_size=board_data['square_size'],
                                 dict_type=cv2.aruco.getPredefinedDictionary(ARUCO_DICT[board_data['dict_type']]))

    def camera_callback(self, input_image):

        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(input_image, desired_encoding="bgr8")

        except CvBridgeError as e:
            print(e)

        self.collect_camera_target_transform()

        # TFPublish.publish_static_stamped_transform(publisher=self.pub_charuco_position,
        #                                            parent_name='cam_wrist',
        #                                            child_name='charuco',
        #                                            transform_stamped=self.transform_memory[-1])
        if len(self.transform_memory) != 0:
            TFPublish.publish_static_stamped_transform(publisher=self.pub_charuco_position,
                                                       parent_name='cam_wrist',
                                                       child_name='charuco',
                                                       transform_stamped=self.transform_memory[-1])

        cv2.imshow('image', self.current_image)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Quit
            print('Shutting down....')
            rospy.signal_shutdown('We are done here')



    def collect_camera_target_transform(self):
        if self.aruco_mode:
            latest_t_vec, latest_r_vec = self.estimate_aruco_pose()
        else:
            self.current_image, latest_r_vec, latest_t_vec = self.arHelper.estimate_charuco_pose(
                image=self.current_image,
                camera_matrix=self.camera_matrix,
                dist_coefficients=self.distortion)

        if latest_t_vec is not None and latest_r_vec is not None:
            latest_r_vec = TypeConverter.rotation_vector_to_quaternions(latest_r_vec)

            latest_t_vec = np.array(latest_t_vec)
            latest_r_vec = np.array(latest_r_vec)
            if not np.isnan(latest_r_vec).any() and not np.isnan(latest_t_vec).any():
                transform = TypeConverter.vectors_to_stamped_transform(translation=latest_t_vec,
                                                                       rotation=latest_r_vec,
                                                                       parent_frame="cam_wrist",
                                                                       child_frame="charuco")
                self.transform_memory.append(transform)
                if len(self.transform_memory) > self.memory_size:
                    self.transform_memory = self.transform_memory[1:]
                
    def estimate_aruco_pose(self):
        marker_size_m = 0.1
        # arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # arucoParams = cv2.aruco.DetectorParameters()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

        aruco_translation = None
        aruco_rotation = None

        if len(corners) > 0:
            for i in range(0, len(ids)):
                rotation_vector, translation_vector, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i],
                                                                                                         marker_size_m,
                                                                                                         self.camera_matrix,
                                                                                                         self.distortion)

                cv2.drawFrameAxes(self.current_image, self.camera_matrix, self.distortion, rotation_vector,
                                  translation_vector,
                                  marker_size_m / 2)
                aruco_translation = translation_vector.flatten()
                # aruco_rotation = TypeConverter.rotation_vector_to_quaternions(rotation_vector.flatten())
        return aruco_translation, rotation_vector

    def get_transform_between(self, origin, to):
        try:
            transform = self.tfBuffer.lookup_transform(origin, to, rospy.Time())

            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print(f"No transform found between {origin} and {to}.")
            return None


if __name__ == '__main__':

    config_file = rospy.get_param(param_name='charuco_position/config')
    path = os.path.join(rospkg.RosPack().get_path('find_and_publish_charuco'),
                        'config/', config_file)
    board_name, camera_name, camera_topic = JSONHelper.get_charuco_info(
        path)
    rospy.init_node('charuco_position')

    charuco_publisher = CharucoPublisher(board_name, camera_name, camera_topic)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()
