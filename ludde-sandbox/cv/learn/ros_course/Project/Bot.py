#!/usr/bin/env python

import cv2
import rospy
from FeatureMatch import FeatureMatch
from PeepDetect import PeepDetect
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from ARDetect import ARDetect


class Bot(object):
    target_height = 0
    target_width = 0
    pd = PeepDetect()
    ad = ARDetect()
    fm = FeatureMatch()
    wanted_image_path = '/home/user/catkin_ws_old/src/opencv_for_robotics_images/Project/Course_images/wanted.png'
    wanted_image = cv2.imread(wanted_image_path, 1)

    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_callback)
        self.bridge_object = CvBridge()

    def camera_callback(self, data):

        try:
            # We select bgr8 because its the OpenCV encoding by default
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
            image = cv_image

            # Find stuff
            self.pd.find_peep(image)
            self.ad.find_tag(image)
            self.fm.find_match(self.wanted_image, image)

            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)


def main():
    load_feature_object = Bot()
    rospy.init_node('bot_node', anonymous=True)

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()