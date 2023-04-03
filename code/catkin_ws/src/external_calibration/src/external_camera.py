#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy

import math
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import tf.transformations as tf

from utils.TFPublish import *


# Broadcast Transform: Base -> Camera
if __name__ == '__main__':
    rospy.init_node('external_camera_node')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    pub_aruco_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('world', 'camera_from_aruco_[0]', rospy.Time())
            publish(trans, pub_aruco_tf)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        publish(publisher=pub_aruco_tf, transform=trans, parent_name="world", child_name="camera_position")
        print(trans)

        rate.sleep()
