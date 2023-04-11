#! /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
#/home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['camera_calibration'],
    package_dir={'': 'src'},
)

setup(**setup_args)
