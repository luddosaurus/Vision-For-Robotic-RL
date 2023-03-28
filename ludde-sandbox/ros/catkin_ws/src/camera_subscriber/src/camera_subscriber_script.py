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


# Y=down, X=right and Z=towards
def pixel_to_camera(u, v, camera_matrix, dist_coeffs):
    # Define the camera matrix and distortion coefficients

    camera_matrix = np.array(camera_matrix)
    dist_coeffs = np.array(dist_coeffs)
    print("camera matrix", camera_matrix.shape)
    print("camera matrix", camera_matrix.dtype)
    print("distortion", dist_coeffs.shape)
    print("distortion", dist_coeffs.dtype)
    # Convert the pixel coordinate (u, v) to a homogeneous 2D point
    # uv_point = np.array([[u, v, 1]], dtype=np.float32).T
    uv_point = np.array([[[u, v]]], dtype=np.float32)
    print("point", uv_point)
    print("point", u)
    print("point", v)
    print("point", uv_point.shape)
    print("point", uv_point.dtype)

    # remove distortion of point
    undistorted_point = cv2.undistortPoints(uv_point, camera_matrix, dist_coeffs)
    x = undistorted_point[0, 0, 0]
    y = undistorted_point[0, 0, 1]

    # convert the undistorted pixel coordinates to homogeneous 2D coordinates
    undistorted_hom = np.array([[x, y, 1]], dtype=np.float32).T

    # compute the homogeneous 3D coordinates X = Pinv*x
    camera_point = np.dot(np.linalg.inv(camera_matrix), undistorted_hom)

    return camera_point


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
            print(centers)
            for u, v in centers:
                print((u,v))
                camera_point = pixel_to_camera(u, v, intrinsic_camera, distortion)
                print(u, " ", v," -> ", camera_point)

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
