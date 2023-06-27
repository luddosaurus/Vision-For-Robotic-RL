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
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import actionlib

from my_robot_msgs.msg import MoveArmAction, MoveArmFeedback, MoveArmResult


class MoveArmActionServer(object):

    def __init__(self):
        self.feedback = MoveArmFeedback()
        self.result = MoveArmResult()
        self.action_server = actionlib.SimpleActionServer('pick_and_place', MoveArmAction, self.goal_callback, False)
        self.action_server.start()

        self.pickup_point_translation = None
        self.place_point_translation = None

    def goal_callback(self, goal):
        self.pickup_point_translation = [goal.pickup_pose.position.x, goal.pickup_pose.position.y,
                                         goal.pickup_pose.position.z]
        self.place_point_translation = [goal.place_pose.position.x, goal.place_pose.position.y,
                                        goal.place_pose.position.z]

        rotation = [1.0, 0., 0., 0.]
        # rotation_90 = [0.7071, 0.7071, 0., 0.]
        # translation = [0.3, -0.2, 0.005]
        # translation_goal = [0.3, 0.2, 0.005]
        # # joint_state_close = [0.01, 0.01]
        # # joint_state_open = [0.03, 0.03]
        joint_state_close = 0.005
        joint_state_open = 0.03
        #
        # # - Translation: [0.268, 0.252, 0.247]
        #
        # # Move the arm to the specified XYZ coordinates
        #
        self.grip(self.pickup_point_translation, rotation, joint_state_close, open_before_move=True)
        # rospy.Rate(1).sleep()
        # move_arm_to_coordinate(translation_ready, rotation)
        # rospy.Rate(1).sleep()
        self.grip(self.place_point_translation, rotation, joint_state_open)
        print('past place')
        # move_arm_to_coordinate(translation_ready, rotation)

        self.result.result = True
        self.action_server.set_succeeded(self.result)

    def create_pose(self, translation, rotation):
        # Create a Pose object
        pose = Pose()

        # Set the translation
        pose.position.x = translation[0]
        pose.position.y = translation[1]
        pose.position.z = translation[2]

        # Set the rotation
        pose.orientation.x = rotation[0]
        pose.orientation.y = rotation[1]
        pose.orientation.z = rotation[2]
        pose.orientation.w = rotation[3]

        return pose

    def grip(self, translation, rotation, joint_state, open_before_move=False):
        if open_before_move:
            client = actionlib.SimpleActionClient('/franka_gripper/gripper_action', GripperCommandAction)

            client.wait_for_server()

            goal = GripperCommandGoal()
            goal.command.position = 0.03
            goal.command.max_effort = 10

            client.send_goal(goal, done_cb=self.mini_goal_callback)

            rospy.Rate(1).sleep()

        translation_elevated_z = translation[:2] + ([translation[2] + 0.1])
        target_pose = self.create_pose(translation_elevated_z, rotation)
        target_pose_grip = self.create_pose(translation, rotation)

        # Initialize ROS node


        # Initialize MoveIt commander
        moveit_commander.roscpp_initialize(sys.argv)

        # Create a RobotCommander instance to get information about the robot
        robot = moveit_commander.RobotCommander()
        # print(robot.get_current_state())
        # # Create a MoveGroupCommander for the arm
        group_name = "panda_manipulator"  # Specify the group name of the arm
        # # group_name = "panda_arm"
        group = moveit_commander.MoveGroupCommander(group_name)
        ready_state = group.get_current_joint_values()
        # Plan and execute the arm movement to the target pose
        group.set_pose_target(target_pose) # approach pose
        group.go(wait=True)
        group.clear_pose_targets()

        group.set_pose_target(target_pose_grip) # grasp pose
        group.go(wait=True)
        group.clear_pose_targets()

        client = actionlib.SimpleActionClient('/franka_gripper/gripper_action', GripperCommandAction)

        client.wait_for_server()

        goal = GripperCommandGoal()
        goal.command.position = joint_state
        goal.command.max_effort = 50

        client.send_goal(goal, done_cb=self.mini_goal_callback)

        rospy.Rate(1).sleep()

        group.set_pose_target(target_pose)
        group.go(wait=True)
        group.clear_pose_targets()

        group.go(ready_state)
        group.clear_pose_targets()
        group.stop()


        # DONT USE THIS
        # # Instantiate a GripperCommander object for the gripper
        # gripper = moveit_commander.MoveGroupCommander("panda_hand")
        # gripper.clear_pose_targets()
        # gripper.stop()
        # # gripper = robot.get_joint('panda_finger_joint1')
        # # Close the gripper to grip the cube
        # gripper.set_joint_value_target(joint_state)  # Specify the gripper closed position
        #
        # gripper.go(wait=True)
        #
        # # gripper.move(gripper.max_bound() * joint_state, True)
        #
        # # Stop the gripper
        # gripper.stop()

    def mini_goal_callback(self, one, two):
        print(f'one: {one}\ntwo: {two}')


# Usage example
if __name__ == '__main__':
    rospy.init_node('move_arm_node', anonymous=True)
    MoveArmActionServer()
    rospy.spin()
# grip_actions()
# # Specify the desired XYZ coordinates
#

# rotation = [1.0, 0., 0., 0.]
# rotation_90 = [0.7071, 0.7071, 0., 0.]
# translation = [0.3, -0.2, 0.005]
# translation_goal = [0.3, 0.2, 0.005]
# # joint_state_close = [0.01, 0.01]
# # joint_state_open = [0.03, 0.03]
# joint_state_close = 0.01
# joint_state_open = 0.03
#
# # - Translation: [0.268, 0.252, 0.247]
#
# # Move the arm to the specified XYZ coordinates
#
# grip(translation, rotation, joint_state_close)
# # rospy.Rate(1).sleep()
# # move_arm_to_coordinate(translation_ready, rotation)
# # rospy.Rate(1).sleep()
# grip(translation_goal, rotation, joint_state_open)
# # move_arm_to_coordinate(translation_ready, rotation)
