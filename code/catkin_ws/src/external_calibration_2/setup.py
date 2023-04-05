#! /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/oskar/somewhere/Thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/lab/somewhere/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['external_calibration_2'],
    package_dir={'': 'src'},
)

setup(**setup_args)
