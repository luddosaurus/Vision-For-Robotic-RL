#! /usr/bin/env python3.8

import os

import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

from camera_calibration.utils.JSONHelper import JSONHelper


class TowerBuilder(object):
    def __init__(self, camera_topics):
        self.current_image_dict = {}
        self.cv_bridge = CvBridge()
        self.camera_topics = camera_topics.keys()

        self.hand_subscriber = rospy.Subscriber(f'{camera_topics[0]}/color/image_raw', Image,
                                                callback=self.callback_hand)
        self.front_subscriber = rospy.Subscriber(f'{camera_topics[1]}/color/image_raw', Image,
                                                 callback=self.callback_front)
        self.top_subscriber = rospy.Subscriber(f'{camera_topics[2]}/color/image_raw', Image, callback=self.callback_top)

        self.hand_depth_subscriber = rospy.Subscriber(f'{camera_topics[0]}/aligned_depth_to_color/image_raw', Image,
                                                      callback=self.callback_hand_depth)
        self.front_depth_subscriber = rospy.Subscriber(f'{camera_topics[1]}/aligned_depth_to_color/image_raw', Image,
                                                       callback=self.callback_front_depth)
        self.top_depth_subscriber = rospy.Subscriber(f'{camera_topics[2]}/aligned_depth_to_color/image_raw', Image,
                                                     callback=self.callback_top_depth)

        self.camera_depth_subscribers = {}
        for camera_topic in camera_topics:
            self.camera_depth_subscribers[camera_topic] = rospy.Subscriber(
                name=f'{camera_topic}/aligned_depth_to_color/image_raw',
                data_class=Image,
                callback=self.camera_depth_callback,
                callback_args=camera_topic
            )

    def callback_hand(self, image):
        topic_name = 'cam_wrist'
        try:

            current_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

            self.current_image_dict[topic_name] = current_image
            self.segment(topic_name=topic_name, current_image=current_image)

        except CvBridgeError as e:
            print(e)

    def callback_front(self, image):
        topic_name = 'cam_front'
        try:

            current_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

            self.current_image_dict[topic_name] = current_image
            self.segment(topic_name=topic_name, current_image=current_image)

        except CvBridgeError as e:
            print(e)

    def callback_top(self, image):
        topic_name = 'cam_top'
        try:

            current_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

            self.current_image_dict[topic_name] = current_image
            self.segment(topic_name=topic_name, current_image=current_image)

        except CvBridgeError as e:
            print(e)

    def callback_hand_depth(self, image):
        topic_name = 'cam_wrist'
        try:

            current_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

            self.current_image_dict[topic_name] = current_image
            self.segment(topic_name=topic_name, current_image=current_image)

        except CvBridgeError as e:
            print(e)

    def callback_front_depth(self, image):
        topic_name = 'cam_front'
        try:

            current_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

            self.current_image_dict[topic_name] = current_image
            self.segment(topic_name=topic_name, current_image=current_image)

        except CvBridgeError as e:
            print(e)

    def callback_top_depth(self, image):
        topic_name = 'cam_top'
        try:

            current_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

            self.current_image_dict[topic_name] = current_image
            self.segment(topic_name=topic_name, current_image=current_image)

        except CvBridgeError as e:
            print(e)

    def segment(self, topic_name, current_image):
        pass


if __name__ == '__main__':
    rospy.init_node('tower_builder_node')

    path = os.path.join(rospkg.RosPack().get_path('object_finder'), 'hsv_exports/')

    config_dict_name = rospy.get_param(param_name='tower_builder_node/config')

    config_file_path = path + config_dict_name + '/hsv'

    parameters = JSONHelper.read_json(config_file_path)

    print(parameters)

    # tower_builder = TowerBuilder(cameras)
