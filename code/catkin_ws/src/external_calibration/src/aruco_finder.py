#! /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/code/venv/bin/python

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
from tf.transformations import quaternion_from_matrix


# Local
from utils.ARHelper import ARHelper
from params.params_remote import *

# Init
arhelper = ARHelper(marker_size_m)
with np.load(calibration_path) as X:
    intrinsic_camera, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef', 'rVector', 'tVector')]

print("ArUcoFinder launched with parameters:")
print(intrinsic_camera, distortion)


# Finds ArUco:s in images and broadcast the tf ArUco to Camera
class ArUcoFinder(object):
    world_points = []
    image_points = []

    def __init__(self):
        self.cv_bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.pub_aruco_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)

    @staticmethod
    def invert_transform(translation, rotation):
        rotation_mat, _ = cv2.Rodrigues(rotation)
        # Change to ArUco to Camera
        inv_rotation = rotation_mat.T
        inv_translation = -np.dot(inv_rotation, translation)
        # Convert to Quaternion
        quaternion = quaternion_from_matrix(rotation_mat)

        return inv_translation, quaternion

    # Publish TF with camera point translation and rotation
    def publish(self, translation, rotation, aruco_id):
        translation, rotation = self.invert_transform(translation, rotation)
        transform_stamped_msg = geometry_msgs.msg.TransformStamped()

        # Info
        transform_stamped_msg.header.stamp = rospy.Time.now()
        transform_stamped_msg.header.frame_id = "ArUco"
        transform_stamped_msg.child_frame_id = "Number " + str(aruco_id)

        # Data
        transform_stamped_msg.transform.translation.x = translation[0]
        transform_stamped_msg.transform.translation.y = translation[1]
        transform_stamped_msg.transform.translation.z = translation[2]
        transform_stamped_msg.transform.rotation.x = rotation[0][0]
        transform_stamped_msg.transform.rotation.y = rotation[1][0]
        transform_stamped_msg.transform.rotation.z = rotation[2][0]
        transform_stamped_msg.transform.rotation.w = rotation[3][0]

        self.pub_aruco_tf.publish(transform_stamped_msg)

    # Finds the ArUco:s location in the camera 3D space
    def callback(self, image):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")

        except CvBridgeError as e:
            print(e)

        # Find ArUco Markers
        image, corners, ids = arhelper.find_markers(image)

        if ids is not None:

            # Find Camera Coordinates 3D
            for index in range(0, len(ids)):
                r_vec, t_vec, obj_corners = cv2.aruco.estimatePoseSingleMarkers(
                    corners=corners,
                    markerLength=marker_size_m,
                    cameraMatrix=intrinsic_camera,
                    distCoeffs=distortion)

                center_point = arhelper.find_center(corners[index], ids[index])
                camera_point = t_vec[index].flatten()
                self.image_points.append(center_point)
                self.world_points.append(camera_point)

            # Display Image
            cv2.imshow('image', image)
            cv2.waitKey(0)


def main():
    rospy.init_node('aruco_finder_node')
    aruco_finder = ArUcoFinder()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
