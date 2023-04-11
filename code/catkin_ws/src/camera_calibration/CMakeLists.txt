cmake_minimum_required(VERSION 3.0.2)
project(camera_calibration)


## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  tf2
  tf2_ros
)

# Add Python Source code
catkin_python_setup()
