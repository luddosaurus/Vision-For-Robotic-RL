#! /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python

import os
import rospkg

marker_size_m = 0.034  # Deprecated

# calibration_path_lab = '/home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/code/catkin_ws/src/camera_calibration' \
#                        '/calibration_data/internal_calibration/MultiMatrix_720.npz'
# calibration_path_remote = '/home/dat14lja/thesis/Vision-For-Robotic-RL/code/catkin_ws/src/camera_calibration' \
#                           '/calibration_data/internal_calibration/MultiMatrix_720.npz'
# calibration_path_laptop = '/home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/code/catkin_ws/src' \
#                           '/camera_calibration' \
#                           '/calibration_data/internal_calibration/MultiMatrix_720.npz'
calibration_path_d455 = os.path.join(rospkg.RosPack().get_path('camera_calibration'),
                                     'calibration_data/internal_calibration/d455_default_MultiMatrix_(800, 1280).npz')
calibration_path_d435 = os.path.join(rospkg.RosPack().get_path('camera_calibration'),
                                     'calibration_data/internal_calibration/MultiMatrix_(1080, 1920).npz')
config_path = os.path.join(rospkg.RosPack().get_path('camera_calibration'), 'config/')

# external_calibration_path = '/home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/code/catkin_ws/src/camera_calibration/calibration_data/external_calibration_data/'
#
#
# external_calibration_path_position = '/home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/code/catkin_ws/src' \
#                             '/camera_calibration/calibration_data/external_calibration_data_position/'

external_calibration_path = os.path.join(rospkg.RosPack().get_path('camera_calibration'),
                                         'calibration_data/external_calibration_data/')

external_calibration_path_position = os.path.join(rospkg.RosPack().get_path('camera_calibration'),
                                                  'calibration_data/external_calibration_data_position/')

# if os.path.exists(calibration_path_lab):
#     calibration_path = calibration_path_lab
# elif os.path.exists(calibration_path_remote):
#     calibration_path = calibration_path_remote
# else:
#     calibration_path = calibration_path_laptop
