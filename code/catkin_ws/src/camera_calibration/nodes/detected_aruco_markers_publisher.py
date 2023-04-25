#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python

import rospy
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Local
from src.camera_calibration.utils.ARHelper import ARHelper
from src.camera_calibration.params.calibration import marker_size_m, calibration_path

# Init
arhelper = ARHelper(marker_size_m)
with np.load(calibration_path) as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]

print("ArUcoFinder launched with parameters:")
print(intrinsic_camera, distortion)


class DetectedArucoPublisher(object):
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.publisher = rospy.Publisher('/detected_aruco_marker_ids', UInt8MultiArray, queue_size=1)
        self.ar_helper = ARHelper(marker_size=marker_size_m)

    def callback(self, image):
        try:
            original_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        except CvBridgeError as e:

            print(e)

        image, _, marker_ids = self.ar_helper.find_markers(original_image)

        if marker_ids is not None:

            data = list()
            for marker_id in marker_ids:
                data.append(marker_id[0])
            marker_ids_message = UInt8MultiArray()
            marker_ids_message.data = data
            self.publisher.publish(marker_ids_message)

        rospy.sleep(0.1)
        # cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(1)


def main():
    rospy.init_node('detected_aruco_markers_node')
    detected_aruco_publisher = DetectedArucoPublisher()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
