#! /usr/bin/python3.8 

# import cv2
# import realsense2_camera

import rospy
from std_msgs.msg import Int32 

rospy.init_node('test_1_node')
pub = rospy.Publisher('/counter', Int32, queue_size=1)
rate = rospy.Rate(2)
count = Int32()
count.data = 0

while not rospy.is_shutdown(): 
  pub.publish(count)
  count.data += 1
  rate.sleep()