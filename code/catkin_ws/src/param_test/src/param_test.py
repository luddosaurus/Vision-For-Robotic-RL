#! /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python

import rospy

if __name__ == '__main__':
    arg_input = rospy.get_param('param_test_node/test')

    print(arg_input)
