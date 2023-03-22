#! /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/ludde-sandbox/venv/bin/python

import cv2

from cv_bridge import CvBridge, CvBridgeError
# from estimate_aruco_pose import estimate_pose
from params import *
import numpy as np
from utils.ARHelper import ARHelper
import rospy
from sensor_msgs.msg import Image


# Init
ah = ARHelper(marker_size_m)
with np.load(calibration_path) as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]
print(intrinsic_camera, distortion)


class RealsenseVideoSubscriber(object):
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)

    def callback(self, image):
        try:
            original_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")

            # image, corners, ids = ah.find_markers(image)
            # if ids is not None:
            #     image = ah.draw_vectors(
            #         img=image,
            #         marker_corners=corners,
            #         marker_ids=ids,
            #         matrix=intrinsic_camera,
            #         distortion=distortion
            #     )

            cv2.imshow('image', original_image)
            cv2.waitKey(0)
        except CvBridgeError as e:
            print(e)
        #  image, distances, centers = estimate_pose(original_image,
        #                                          cv2.aruco.DICT_4X4_50,
        #                                          intrinsic_camera, distortion, marker_size_m)


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
