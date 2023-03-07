#! /usr/bin/python3.8 

import cv2

from cv_bridge import CvBridge, CvBridgeError

import rospy
from sensor_msgs.msg import Image

class RealsenseVideoSubscriber(object):
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/realsense_camera', Image, self.callback)

    def callback(self, image):
        try:
            original_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        cv2.imshow('image', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        
def main():
    rospy.init_node('realsense_video_topic_subscriber_node')
    realsense_video_subscriber = RealsenseVideoSubscriber()

    try:
        rospy.spin()
    except KeyboardInterrupt: 
        print('Shutting down.')
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()