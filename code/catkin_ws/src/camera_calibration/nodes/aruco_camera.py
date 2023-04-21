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
from camera_calibration.utils.TFPublish import publish_transform
from camera_calibration.utils.MathHelper import invert_transform, \
    rotation_vector_to_quaternions, riemannian_mean
from camera_calibration.params.calibration import marker_size_m, calibration_path
from camera_calibration.utils.TFTransformer import TFTransformer

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
        self.transforms = {"aruco_[0]": list(), "aruco_[1]": list()}
        self.use_charuco = False

    # Finds the ArUco:s location in the camera 3D space
    def callback(self, image):

        try:
            image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

        except CvBridgeError as e:
            print(e)


        # Find ArUco Markers
        image, corners, ids = arhelper.find_markers(image)

        # image = ARHelper.draw_vectors(image, corners, ids, intrinsic_camera, distortion)

        if ids is not None:

            # todo make sure it's the right id / marker_size (if we use different for table)
            
            # Find Camera Coordinates 3D
            if self.use_charuco: 
                r_vecs, t_vecs = ARHelper.estimate_charuco_pose(
                    image=image,
                    camera_matrix=intrinsic_camera,
                    dist_coefficients=distortion)
            else:
                r_vecs, t_vecs, obj_corners = cv2.aruco.estimatePoseSingleMarkers(
                    corners=corners,
                    markerLength=marker_size_m,
                    cameraMatrix=intrinsic_camera,
                    distCoeffs=distortion)
            
            # NEW
            ARHelper.draw_vectors(image, intrinsic_camera, distortion, r_vecs, t_vecs)

            for aruco_id, rotation, translation, corner_points in zip(ids, r_vecs, t_vecs, corners):

                # change to aruco to camera
                translation, rotation = invert_transform(translation, rotation)

                transform_name = f"aruco_to_camera_{aruco_id}"
                aruco_name = f"aruco_{aruco_id}"
                if aruco_name in self.transforms.keys():
                    self.transforms[aruco_name].append((translation, rotation))
                else:
                    self.transforms[aruco_name] = [(translation, rotation)]

                if len(self.transforms[aruco_name]) > 30:
                    translation, rotation = self.create_average_transform(aruco_name, aruco_name, transform_name)
                    if not np.isnan(translation).any():
                        publish_transform(
                            publisher=self.pub_aruco_tf,
                            parent_name=aruco_name,
                            child_name=transform_name,
                            translation=translation,
                            rotation=rotation
                        )
                    self.transforms = dict()

        # Display Image
        cv2.imshow('image', image)
        cv2.waitKey(1)


    def create_average_transform(self, aruco_name, parent_frame, child_frame):
        transformations = list()
        for transform in self.transforms[aruco_name]:
            transformations.append(
                TFTransformer.vectors_to_stamped_transform(transform[0], transform[1], parent_frame, child_frame))

        return riemannian_mean(transformations)


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
