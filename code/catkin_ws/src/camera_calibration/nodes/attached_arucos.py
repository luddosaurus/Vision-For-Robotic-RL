#! /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python

import rospy
import tf2_ros

from camera_calibration.params.panda_hand_aruco import *
from camera_calibration.utils.TFPublish import publish_static_transform


class AttachedArucoBroadcaster:

    def __init__(self):

        self.pub_tf_static = tf2_ros.StaticTransformBroadcaster()

        #
    def spawn_static_aruco_markers(self, parent_frame_name, aruco_list):
        for aruco in aruco_list:
            publish_static_transform(
                publisher=self.pub_tf_static,
                parent_name=parent_frame_name,
                child_name=aruco.get_name(),
                translation=aruco.translation,
                rotation=aruco.rotation)


if __name__ == '__main__':
    rospy.init_node('attached_aruco_broadcaster')
    tfb = AttachedArucoBroadcaster()
    while not rospy.is_shutdown():
        rospy.sleep(0.1)
        tfb.spawn_static_aruco_markers(arm_parent_frame, arm_arucos)
        tfb.spawn_static_aruco_markers(table_parent_frame, table_arucos)

    rospy.spin()
