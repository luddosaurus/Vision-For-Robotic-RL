#! /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python

import os

marker_size_m = 0.034
calibration_path_lab = '/home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/code/catkin_ws/src' \
                       '/camera_calibration/src/intrinsic_camera_calibration_data/MultiMatrix_720.npz'
calibration_path_remote = '/home/dat14lja/thesis/Vision-For-Robotic-RL/code/catkin_ws/src/camera_calibration/src' \
                          '/intrinsic_camera_calibration_data/MultiMatrix_720.npz'
calibration_path_laptop = '/home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/code/catkin_ws/src' \
                          '/camera_calibration/src/intrinsic_camera_calibration_data/MultiMatrix_720.npz'
calibration_path = ""

if os.path.exists(calibration_path_lab):
    calibration_path = calibration_path_lab
elif os.path.exists(calibration_path_remote):
    calibration_path = calibration_path_remote
else:
    calibration_path = calibration_path_laptop
