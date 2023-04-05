#! /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
from external_calibration_2.model.my_aruco import ArUco

# Params for Panda Hand to attached ArUco Offset


frame_name = "panda_hand"

# Offset left ArUco
left_name = "aruco_[0]"
left_x = 0.036
left_y = 0.055
left_z = 0.04

left_rotation_w = 0.5
left_rotation_x = -0.5
left_rotation_y = 0.5
left_rotation_z = -0.5

# Offset right ArUco
right_name = "aruco_[1]"
right_x = 0.036
right_y = -0.055
right_z = 0.04

right_rotation_w = 0.5
right_rotation_x = -0.5
right_rotation_y = 0.5
right_rotation_z = -0.5

arm_arucos = [ArUco(aruco_id=0,
                    translation=(0.036, 0.055, 0.04),
                    rotation=(0.5, -0.5, 0.5, -0.5)
                    ),
              ArUco(
                  aruco_id=1,
                  translation=(0.036, -0.055, 0.04),
                  rotation=(0.5, -0.5, 0.5, -0.5)
              )
              ]

[0.446, -0.304, 0.037]

table_arucos = [ArUco(
    aruco_id=6,
    translation=(0.446, -0.304, -0.015),
    rotation=(0.7071, 0.0, 0.0, 0.7071)
), ArUco(
    aruco_id=7,
    translation=(0.443, -0.407, -0.015),
    rotation=(0.7071, 0.0, 0.0, 0.7071)
), ArUco(
    aruco_id=8,
    translation=(0.154, -0.307, -0.015),
    rotation=(0.7071, 0.0, 0.0, 0.7071)
), ArUco(
    aruco_id=9,
    translation=(0.152, 404, -0.015),
    rotation=(0.7071, 0.0, 0.0, 0.7071)
), ]
