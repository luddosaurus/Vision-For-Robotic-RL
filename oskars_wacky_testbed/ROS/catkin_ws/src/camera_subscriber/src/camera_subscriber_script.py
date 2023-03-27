#! /usr/bin/python3.8

import cv2

from cv_bridge import CvBridge, CvBridgeError
from estimate_aruco_pose import estimate_pose, transform_aruco_to_camera
import numpy as np

import rospy
from sensor_msgs.msg import Image

with np.load(
        '/home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/oskars_wacky_testbed/ROS/catkin_ws/src/camera_subscriber/src/calib_data_2_new/MultiMatrix.npz') as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]
print(intrinsic_camera, distortion)

marker_size_m = 0.034


class RealsenseVideoSubscriber(object):
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)

    def callback(self, image):
        try:
            original_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        transform_aruco_to_camera(original_image, cv2.aruco.DICT_4X4_50,
                                  intrinsic_camera, distortion, marker_size_m)
        image, distances, centers = estimate_pose(original_image,
                                                  cv2.aruco.DICT_4X4_50,
                                                  intrinsic_camera, distortion, marker_size_m)

        cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


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
