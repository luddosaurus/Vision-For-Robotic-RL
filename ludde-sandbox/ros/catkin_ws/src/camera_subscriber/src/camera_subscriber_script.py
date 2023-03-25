#! /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/ludde-sandbox/venv/bin/python

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import rospy
from sensor_msgs.msg import Image

from utils.ARHelper import ARHelper
from params_remote import *
from utils.photoshop import *

# Init
arhelper = ARHelper(marker_size_m)
with np.load(calibration_path) as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]
print("Camera Subscriber launched with parameters:")
print(intrinsic_camera, distortion)


class RealsenseVideoSubscriber(object):
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)

    def callback(self, image):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")

        except CvBridgeError as e:
            print(e)

        # Find ArUco Markers
        image, corners, ids = arhelper.find_markers(image)

        # Draw Vectors
        if ids is not None:
            # image = draw_vectors(
            #     img=image,
            #     marker_corners=corners,
            #     marker_ids=ids,
            #     matrix=intrinsic_camera,
            #     distortion=distortion
            # )

            centers = arhelper.find_center(corners, ids)
            paint_dots(image, centers)

        # Display Image
        cv2.imshow('image', image)
        cv2.waitKey(0)


def main():
    rospy.init_node('camera_subscriber_node')
    realsense_video_subscriber = RealsenseVideoSubscriber()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
