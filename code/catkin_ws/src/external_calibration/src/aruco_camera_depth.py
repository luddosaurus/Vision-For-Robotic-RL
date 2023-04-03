#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/oskar/somewhere/Thesis/Vision-For-Robotic-RL/code/venv/bin/python
# /home/lab/somewhere/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

# Standard
import numpy as np

# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS
import rospy
from sensor_msgs.msg import Image
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import tf.transformations as tf

# Local
from utils.ARHelper import ARHelper
from params.calibration_remote import *
from utils.TFPublish import *
from utils.MathHelper import *

# Init
arhelper = ARHelper(marker_size_m)
with np.load(calibration_path) as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]

print("ArUcoFinder launched with parameters:")
print(intrinsic_camera, distortion)


# Finds ArUco:s in images and broadcast the tf ArUco to Camera
class ArUcoFinder(object):

    def __init__(self):
        self.cv_bridge = CvBridge()
        self.subscriber_image = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.subscriber_depth = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.callback)
        self.pub_aruco_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)

    # Finds the ArUco:s location in the camera 3D space
    def callback(self, image):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
            # todo get depth callback here as well? or only aligned_depth?
        except CvBridgeError as e:
            print(e)

        # Find ArUco Markers
        image, corners, ids = arhelper.find_markers(image)

        if ids is not None:

            # Find Camera Coordinates 3D

            # Returns the center points of the aruco:s
            r_vecs, t_vecs, obj_corners = cv2.aruco.estimatePoseSingleMarkers(
                corners=corners,
                markerLength=marker_size_m,
                cameraMatrix=intrinsic_camera,
                distCoeffs=distortion)

            # todo change z coord to the one in depth here

            for aruco_id, rotation, translation, corner_points in zip(ids, r_vecs, t_vecs, corners):

                center_point = arhelper.find_center(corner_points, aruco_id)
                camera_point = translation.flatten()

                # change to aruco to camera
                translation, rotation = invert_transform(translation, rotation)

                publish(
                    publisher=self.pub_aruco_tf,
                    parent_name=f"aruco_{aruco_id}",
                    child_name=f"camera_from_aruco_{aruco_id}",
                    translation=translation,
                    rotation=rotation
                )
            # Display Image
            cv2.imshow('image', image)
            cv2.waitKey(1)


def main():
    rospy.init_node('aruco_camera_node')
    aruco_finder = ArUcoFinder()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
