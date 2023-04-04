#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
from params.panda_hand_aruco import *


class ArucoHandBroadcaster:

    def __init__(self):
        self.pub_tf_left = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)
        self.pub_tf_right = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)

        while not rospy.is_shutdown():
            # Run this loop at about 10Hz
            rospy.sleep(0.1)

            t_left = geometry_msgs.msg.TransformStamped()
            t_left.header.frame_id = frame_name
            t_left.header.stamp = rospy.Time.now()
            t_left.child_frame_id = left_name
            t_left.transform.translation.x = left_x
            t_left.transform.translation.y = left_y
            t_left.transform.translation.z = left_z

            t_left.transform.rotation.x = left_rotation_x
            t_left.transform.rotation.y = left_rotation_y
            t_left.transform.rotation.z = left_rotation_z
            t_left.transform.rotation.w = left_rotation_w

            t_right = geometry_msgs.msg.TransformStamped()
            t_right.header.frame_id = frame_name
            t_right.header.stamp = rospy.Time.now()
            t_right.child_frame_id = right_name
            t_right.transform.translation.x = right_x
            t_right.transform.translation.y = right_y
            t_right.transform.translation.z = right_z

            t_right.transform.rotation.x = right_rotation_x
            t_right.transform.rotation.y = right_rotation_y
            t_right.transform.rotation.z = right_rotation_z
            t_right.transform.rotation.w = right_rotation_w

            tfm_left = tf2_msgs.msg.TFMessage([t_left])
            tfm_right = tf2_msgs.msg.TFMessage([t_right])
            self.pub_tf_left.publish(tfm_left)
            self.pub_tf_right.publish(tfm_right)


if __name__ == '__main__':
    rospy.init_node('aruco_hand_node')
    tfb = ArucoHandBroadcaster()

    rospy.spin()
