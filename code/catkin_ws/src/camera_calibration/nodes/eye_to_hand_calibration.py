#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy
from camera_calibration.utils.TFPublish import *
import geometry_msgs
from std_msgs.msg import UInt8MultiArray
import tf
import tf2_ros
from tf.transformations import quaternion_matrix
import numpy as np
from time import time


class EyeToHandEstimator(object):
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.pub_aruco_tf = tf2_ros.StaticTransformBroadcaster()
        self.marker_subscriber = rospy.Subscriber('/detected_aruco_marker_ids', UInt8MultiArray, self.marker_callback)
        self.transformations = list()
        self.start_time = time()
        # todo
        # Get world-ee frames
        # Get camera-aruco frames
        # Sync frames
        # AX = XB ish

    def marker_callback(self, message):
        pass

if __name__ == '__main__':
    rospy.init_node('hand_eye_node')
    hand_eye_estimator = EyeToHandEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
