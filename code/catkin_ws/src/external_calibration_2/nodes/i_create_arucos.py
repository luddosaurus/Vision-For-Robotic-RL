#! /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

from external_calibration_2.model import *


def main():
    ar = my_aruco.ArUco(aruco_id=0,
                        translation=(0.036, 0.055, 0.04),
                        rotation=(0.5, -0.5, 0.5, -0.5)
                        )

    ar.say_hello()


if __name__ == '__main__':
    main()
