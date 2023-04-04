import os

marker_size_m = 0.034
calibration_path_lab = '/home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/code/catkin_ws/src/external_calibration/src/calibration_data/MultiMatrix.npz'
calibration_path_remote = '/home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/catkin_ws/src' \
                          '/external_calibration/src/calibration_data/MultiMatrix.npz'
calibration_path = calibration_path_lab if os.path.exists(calibration_path_lab) else calibration_path_remote

