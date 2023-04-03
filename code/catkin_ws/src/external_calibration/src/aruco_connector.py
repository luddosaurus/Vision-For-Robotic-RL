#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy

import math
import tf2_ros
from geometry_msgs.msg import TransformStamped


# Attach ArUco:s with static transform
if __name__ == '__main__':
    rospy.init_node('aruco_connector_node')
    static_broadcaster = tf2_ros.StaticTransformBroadcaster()

    # Static transform between ArUco on arm and Aruco in camera
    # todo unclear
    static_transform = TransformStamped()
    static_transform.header.stamp = rospy.Time.now()
    static_transform.header.frame_id = "left_aruco"
    static_transform.child_frame_id = "number_0"

    # Offset 0
    static_transform.transform.translation.x = 0.0
    static_transform.transform.translation.y = 0.0
    static_transform.transform.translation.z = 0.0
    static_transform.transform.rotation.x = 0.0
    static_transform.transform.rotation.y = 0.0
    static_transform.transform.rotation.z = 0.0
    static_transform.transform.rotation.w = 1.0

    # Publish the static transform
    static_broadcaster.sendTransform(static_transform)

    rospy.spin()