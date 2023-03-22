#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv2


def main():

    rospy.init_node('filter_node', anonymous=True)
    img = cv2.imread('/home/user/catkin_ws_old/src/opencv_for_robotics_images/Unit_2/Course_images/test_img_b.jpg')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(450,350))
    cv2.imshow('Origin', img)

    #Apply the horizontal sobel operator with a kernel size of 3
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    cv2.imshow('X',sobelx)

    #Apply the vertical sobel operator with a kernel size of 3
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    cv2.imshow('Y',sobely)

    cv2.waitKey(0)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
