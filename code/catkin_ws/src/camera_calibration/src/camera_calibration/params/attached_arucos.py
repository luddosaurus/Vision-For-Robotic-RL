#! /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python

from camera_calibration.model.aruco import ArUco

# Params for ArUco Offset in meters

arm_parent_frame = "panda_hand"
arm_aruco_size = 0.034
arm_arucos = [
    ArUco(
        aruco_id=0,
        translation=(0.036, 0.055, 0.04),
        rotation=(0.5, -0.5, 0.5, -0.5)
    ),
    ArUco(
        aruco_id=1,
        translation=(0.036, -0.055, 0.04),
        rotation=(0.5, -0.5, 0.5, -0.5)
    )
]

table_parent_frame = "world"
table_aruco_size = 0.034
table_arucos = [ArUco(
    aruco_id=6,
    translation=(0.446, -0.304, -0.015),
    rotation=(0.0, 0.0, 0.7071, 0.7071)
), ArUco(
    aruco_id=7,
    translation=(0.443, 0.407, -0.015),
    rotation=(0.0, 0.0, 0.7071, 0.7071)
), ArUco(
    aruco_id=8,
    translation=(0.154, -0.307, -0.015),
    rotation=(0.0, 0.0, 0.7071, 0.7071)
), ArUco(
    aruco_id=9,
    translation=(0.152, 0.404, -0.015),
    rotation=(0.0, 0.0, 0.7071, 0.7071)
), ]
