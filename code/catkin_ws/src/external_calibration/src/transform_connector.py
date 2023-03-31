#! /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy

import math
import tf2_ros

if __name__ == '__main__':
    rospy.init_node('tf2_listener')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)


    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            # trans = tfBuffer.lookup_transform(turtle_name, 'turtle1', rospy.Time())
            trans = tfBuffer.lookup_transform('panda_hand', 'left_aruco', rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        print(trans)

        rate.sleep()