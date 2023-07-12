#! usr/bin/env python3.8

import os

import rospkg
import rospy

from camera_calibration.utils.JSONHelper import JSONHelper

if __name__ == '__main__':
    rospy.init_node('tower_builder_node')

    path = os.path.join(rospkg.RosPack().get_path('tower_builder'), 'config/')

    config_file_name = rospy.get_param(param_name='tower_builder/config')

    config_file_path = path + config_file_name

    parameters = JSONHelper.read_json(config_file_path)

    cameras = parameters['cameras']
    camera_topics = cameras.keys()
    print(camera_topics)
