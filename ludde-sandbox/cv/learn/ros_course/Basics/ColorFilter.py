#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np


class FilterImage(object):

    def __init__(self):

        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.camera_callback)
        self.bridge_object = CvBridge()

    def filter_image(self, name, image, min_limit, max_limit):
        image = cv.resize(image, (300, 300))
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, min_limit, max_limit)
        res = cv.bitwise_and(image, image, mask=mask)
        cv.imshow(name, res)

    def camera_callback(self, data):
        try:
            # We select bgr8 because its the OpenCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(
                data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        # FIlter
        min_green = np.array([50, 220, 220])
        max_green = np.array([60, 255, 255])

        min_red = np.array([0, 255*0.65, 245])
        max_red = np.array([5, 255*0.75, 255])

        min_blue = np.array([50, 70, 220])
        max_blue = np.array([90, 100, 255])

        self.filter_image('RED', cv_image, min_red, max_red)
        self.filter_image('BLUE', cv_image, min_blue, max_blue)
        self.filter_image('Green', cv_image, min_green, max_green)

        cv.imshow('image', cv_image)

        cv.waitKey(0)


def main():
    rospy.init_node('color_node', anonymous=True)
    showing_image_object = FilterImage()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
