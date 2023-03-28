#! /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/ludde-sandbox/venv/bin/python

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def plot_3d_points_and_vector(points, vector):
    # create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the 3D points
    points = np.array(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, c='r', marker='o')

    # plot the 3D vector
    p, v = vector
    v = np.array(v)
    v /= np.linalg.norm(v)  # normalize the vector
    ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], length=1.0, color='g')

    # set the limits of the plot
    max_range = np.array([np.max(x) - np.min(x), np.max(y) - np.min(y), np.max(z) - np.min(z)]).max()
    mid_x = (np.max(x) + np.min(x)) * 0.5
    mid_y = (np.max(y) + np.min(y)) * 0.5
    mid_z = (np.max(z) + np.min(z)) * 0.5
    # ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    # ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    # ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    # add labels to the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # show the plot
    plt.show()


def get_camera_pose(camera_matrix, distortion_coefficients, image_points, world_points):

    image_points = np.array(image_points, dtype=np.float64)
    world_points = np.array(world_points, dtype=np.float64)
    print("image\n", image_points.dtype)
    print("world\n", world_points.dtype)

    # undistort image points
    # image_points = cv2.undistortPoints(image_points,
    #                                              camera_matrix,
    #                                              distortion_coefficients)

    # estimate camera pose
    retval, rvec, tvec = cv2.solvePnP(world_points,
                                      image_points,
                                      camera_matrix,
                                      distortion_coefficients)

    # convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # calculate viewing direction of camera
    view_dir = -R[:, 2]

    print("Camera pose:")
    print("Rotation matrix:")
    print(R)
    print("Translation vector:")
    print(tvec)
    print("Viewing direction:")
    print(view_dir)
    return tvec, rvec


class RealsenseVideoSubscriber(object):
    world_points = []
    image_points = []

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
        #if ids is not None:

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

            # paint_dots(image, center_points)
            # for image_point in center_points:
            #     self.world_points.append(camera_point)
            #     self.image_points.append(image_point)
            #     print(image_point, " -> \n", camera_point)

        # Display Image
        cv2.imshow('image', image)
        cv2.waitKey(0)

        # todo get camera pose
        if len(self.world_points) > 10:
            camera_pose = get_camera_pose(intrinsic_camera,
                                          distortion,
                                          self.image_points,
                                          self.world_points)

            plot_3d_points_and_vector(self.world_points, camera_pose)

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
