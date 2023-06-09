#! /usr/bin/env python3.8
# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose

def move_arm_to_coordinate(x, y, z):
    # Initialize ROS node
    rospy.init_node('move_arm_node', anonymous=True)

    # Initialize MoveIt commander
    moveit_commander.roscpp_initialize(sys.argv)

    # Create a RobotCommander instance to get information about the robot
    robot = moveit_commander.RobotCommander()

    # Create a MoveGroupCommander for the arm
    group_name = "panda_arm"  # Specify the group name of the arm
    group = moveit_commander.MoveGroupCommander(group_name)

    # Set the target pose
    target_pose = Pose()
    target_pose.position.x = x
    target_pose.position.y = y
    target_pose.position.z = z
    target_pose.orientation.w = 1.0  # Assuming a default orientation

    # Set the target pose for the arm
    group.set_pose_target(target_pose)

    # Plan and execute the motion
    plan = group.go(wait=True)

    # Stop the MoveGroup interface after completing the motion
    moveit_commander.roscpp_shutdown()

    # Exit the ROS node
    rospy.signal_shutdown("Movement completed.")


# Usage example
if __name__ == '__main__':
    # Specify the desired XYZ coordinates
    x = 0.5
    y = 0.3
    z = 0.2

    # Move the arm to the specified XYZ coordinates
    move_arm_to_coordinate(x, y, z)