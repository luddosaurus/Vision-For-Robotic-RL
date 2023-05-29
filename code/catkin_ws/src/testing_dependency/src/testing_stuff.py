#! usr/bin/Python3.8

import rospy
from std_msgs.msg import String

if __name__ == '__main__':
    rospy.init_node('test_node')
    publisher = rospy.Publisher('/i_am_alive', String, queue_size=10)
    while not rospy.is_shutdown():
        publisher.publish(String("Totally Alive"))
        rospy.Rate(1).sleep()
