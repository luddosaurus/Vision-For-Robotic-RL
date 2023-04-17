#! /home/csproj_vision/PycharmProjects/Vision-For-Robotic-RL/venv3/bin/python
# /home/oskarlarsson/PycharmProjects/Vision-For-Robotic-RL/venv/bin/python
# /home/dat14lja/thesis/Vision-For-Robotic-RL/code/venv/bin/python




# Standard
import numpy as np

# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS
import rospy
from sensor_msgs.msg import Image
import tf2_msgs.msg
import geometry_msgs.msg
import tf.transformations as tf

# Local
from camera_calibration.utils.ARHelper import ARHelper
from camera_calibration.utils.TFPublish import *
from camera_calibration.utils.MathHelper import *
from camera_calibration.params.calibration import marker_size_m, calibration_path

# Init
arhelper = ARHelper(marker_size_m)
with np.load(calibration_path) as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]

print("ArUcoFinder launched with parameters:")
print(intrinsic_camera, distortion)


# Finds ArUco:s from images and broadcast the tf (ArUco to Camera)
class ArUcoFinder(object):

    def __init__(self):
        self.cv_bridge = CvBridge()
        # todo add depth here
        self.subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.pub_aruco_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)

    # Finds the ArUco:s location in the camera 3D space
    def callback(self, image):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")

        except CvBridgeError as e:
            print(e)

        # Find ArUco Markers
        image, corners, ids = arhelper.find_markers(image)

        if ids is not None:

            # todo make sure it's the right id / marker_size

            # Find Camera Coordinates 3D
            r_vecs, t_vecs, obj_corners = cv2.aruco.estimatePoseSingleMarkers(
                corners=corners,
                markerLength=marker_size_m,
                cameraMatrix=intrinsic_camera,
                distCoeffs=distortion)

            for aruco_id, rotation, translation, corner_points in zip(ids, r_vecs, t_vecs, corners):
                center_point = arhelper.find_center(corner_points, aruco_id)
                camera_point = translation.flatten()

                # change to aruco to camera
                translation, rotation = invert_transform(translation, rotation)

                publish_transform(
                    publisher=self.pub_aruco_tf,
                    parent_name=f"aruco_{aruco_id}",
                    child_name=f"camera_from_aruco_{aruco_id}",
                    translation=translation,
                    rotation=rotation
                )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
