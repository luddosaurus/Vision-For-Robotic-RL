#! /usr/bin/env python3.8

import rospy
import cv2
import numpy as np

from camera_calibration.params.calibration import calibration_path_d455, calibration_path_d435
from camera_calibration.utils.TypeConverter import TypeConverter
from camera_calibration.utils.TFPublish import TFPublish
from camera_calibration.utils.JSONHelper import JSONHelper

from utils.DaVinci import DaVinci
from utils.ColorObjectFinder import ColorObjectFinder

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

import tf2_ros


# todo
# add finding system
# get image from topic aligned with depth
# get camera pose (world)
# get pose of object (camera) -> (world
# move arm to object


class ObjectFinder:

    def __init__(self, intrinsic_matrix):
        self.pose_estimate = True
        self.intrinsic_matrix = intrinsic_matrix
        self.cof = ColorObjectFinder()
        self.cv_bridge = CvBridge()

        # todo get camera pose in world frame
        self.window = 'ColorDetection'
        self.gui_created = False
        self.start_state = self.cof.get_state()
        self.current_image = None

        self.camera_subscriber = rospy.Subscriber(
            '/camera/color/image_raw',
            Image, self.camera_color_callback)

        self.aligned_depth_subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image,
                                                         self.camera_depth_aligned_callback)

        self.center_x = None
        self.center_y = None
        self.center_z = None

        self.center_broadcaster = tf2_ros.StaticTransformBroadcaster()

    def create_layout(self):
        cv2.namedWindow(self.window)

        cv2.createTrackbar("Hue", self.window,
                           self.start_state[self.cof.HUE], self.cof.HUE_MAX,
                           lambda value: self.cof.update_value(value, self.cof.HUE))
        cv2.createTrackbar("Saturation", self.window,
                           self.start_state[self.cof.SATURATION], self.cof.SAT_MAX,
                           lambda value: self.cof.update_value(value, self.cof.SATURATION))
        cv2.createTrackbar("Value", self.window,
                           self.start_state[self.cof.VALUE], self.cof.VAL_MAX,
                           lambda value: self.cof.update_value(value, self.cof.VALUE))

        cv2.createTrackbar("Hue Margin", self.window,
                           self.start_state[self.cof.HUE_MARGIN], self.cof.HUE_MAX,
                           lambda value: self.cof.update_value(value, self.cof.HUE_MARGIN))
        cv2.createTrackbar("Sat Margin", self.window,
                           self.start_state[self.cof.SATURATION_MARGIN], self.cof.SAT_MAX,
                           lambda value: self.cof.update_value(value, self.cof.SATURATION_MARGIN))
        cv2.createTrackbar("Val Margin", self.window,
                           self.start_state[self.cof.VALUE_MARGIN], self.cof.VAL_MAX,
                           lambda value: self.cof.update_value(value, self.cof.VALUE_MARGIN))

        cv2.createTrackbar("Noise", self.window,
                           self.start_state[self.cof.NOISE], self.cof.NOISE_MAX,
                           lambda value: self.cof.update_value(value, self.cof.NOISE))
        cv2.createTrackbar("Fill", self.window,
                           self.start_state[self.cof.FILL], self.cof.FILL_MAX,
                           lambda value: self.cof.update_value(value, self.cof.FILL))

        cv2.setMouseCallback(self.window, self.click)

    def update_trackbars(self):
        current_state = self.cof.get_state()
        cv2.setTrackbarPos("Hue", self.window, current_state[self.cof.HUE])
        cv2.setTrackbarPos("Saturation", self.window, current_state[self.cof.SATURATION])
        cv2.setTrackbarPos("Value", self.window, current_state[self.cof.VALUE])
        cv2.setTrackbarPos("Hue Margin", self.window, current_state[self.cof.HUE_MARGIN])
        cv2.setTrackbarPos("Sat Margin", self.window, current_state[self.cof.SATURATION_MARGIN])
        cv2.setTrackbarPos("Val Margin", self.window, current_state[self.cof.VALUE_MARGIN])
        cv2.setTrackbarPos("Noise", self.window, current_state[self.cof.NOISE])
        cv2.setTrackbarPos("Fill", self.window, current_state[self.cof.FILL])

    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("click!")
            self.cof.set_image_coordinate_color(self.current_image, x, y)
            self.update_trackbars()

    def camera_depth_aligned_callback(self, aligned_depth):
        # print(aligned_depth)
        aligned_input_depth = None
        try:
            aligned_input_depth = self.cv_bridge.imgmsg_to_cv2(
                aligned_depth, desired_encoding="passthrough")

        except CvBridgeError as e:
            print(e)

        # print(aligned_input_depth[a])
        # Find 3D point
        # cv2.imshow('test', aligned_input_depth)
        if self.pose_estimate and self.center_x is not None and aligned_input_depth is not None:
            depth_array = np.array(aligned_input_depth, dtype=np.float32)
            # print(depth_array.shape)
            if self.center_x <= depth_array.shape[1] and self.center_y <= depth_array.shape[0]:
                # print(self.center_x, depth_array.shape[1])
                depth = depth_array[self.center_y][self.center_x] / 1000
                print('center depth:', depth)

                # todo find depth of coordinate (x,y)
                position = self.cof.pixel_to_3d_coordinate((self.center_x, self.center_y), depth, self.intrinsic_matrix)
                # print(position)
                pose_info = f"x{position[0]:.2f} : y{position[1]:.2f}, z{position[2]:.2f}"

                print(pose_info)
                self.center_z = position[2]
                self.position = position
                self.broadcast_point()

                # todo convert to world frame
                # todo publish

    def broadcast_point(self):
        # transform = TypeConverter.vectors_to_stamped_transform(translation=[self.x, self.y, self.z],
        #                                                        rotation=[0, 0, 0, 0, 0],
        #                                                        parent_frame='camera_estimateTSAI', child_frame='cube')
        # self.center_broadcaster.sendTransform
        # TFPublish.publish_transform(publisher=self.center_broadcaster, rotation=[0, 0, 0, 0, 0],
        #                             translation=[self.center_x, self.center_y, self.center_z], parent_name='camera_estimate0',
        #                             child_name='cube')
        TFPublish.publish_static_transform(publisher=self.center_broadcaster,
                                           parent_name='camera_estimate0',
                                           child_name=f'cube',
                                           rotation=[0., 0., 0., 1.],
                                           translation=self.position)

    def camera_color_callback(self, input_image):

        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(
                input_image, desired_encoding="bgr8")

        except CvBridgeError as e:
            print(e)

        if not self.gui_created:
            self.create_layout()
            self.gui_created = True
        image = self.current_image

        # Mask
        mask_image = self.cof.get_hsv_mask(image=image)
        res = cv2.bitwise_and(image, image, mask=mask_image)
        mask = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

        # Find center
        self.center_x, self.center_y = self.cof.find_mask_center(mask_image)
        pose_info = ""
        if self.center_x is not None:
            self.cof.draw_dot(res, self.center_x, self.center_y)

        # Show Image
        stacked = np.hstack((image, res))

        info = "[0-9] states, [m]ove to, [q]uit"
        DaVinci.draw_text_box(
            image=stacked,
            text=info
        )

        slot_info = f"Color State [{self.cof.current_state_index}]"
        DaVinci.draw_text_box(
            image=stacked,
            text=slot_info,
            position="top_left"
        )

        if self.pose_estimate and pose_info != "":
            DaVinci.draw_text_box(
                image=stacked,
                text=pose_info,
                position="top_right"
            )

        scale = 0.5
        cv2.imshow(self.window, cv2.resize(stacked, None, fx=scale, fy=scale))

        # Input
        key = cv2.waitKey(1) & 0xFF
        key_str = chr(key)

        if key_str.isdigit() and 0 <= int(key_str) <= 9:
            key_number = int(key_str)
            self.cof.current_state_index = key_number
            self.update_trackbars()
            print(f"Switching to state {key_number}")

        elif key == ord('m'):
            print("Going to...")

        elif key == ord('q'):
            rospy.signal_shutdown('Bye :)')


def load_intrinsics(eye_in_hand):
    camera_intrinsics = JSONHelper.get_camera_intrinsics('d435_480p_testing')
    # board_data = JSONHelper.get_board_parameters(board_name)

    camera_matrix = np.array(camera_intrinsics['camera_matrix'])
    distortion = np.array(camera_intrinsics['distortion'])

    # with np.load(calibration_path_d435 if eye_in_hand else calibration_path_d455) as X:
    #     intrinsic, distortion, _, _ = [X[i] for i in ('camMatrix', 'distCoef',
    #                                                   'rVector', 'tVector')]
    # print("ArUcoFinder launched with internal parameters:")
    print(camera_matrix, distortion)
    return camera_matrix


if __name__ == '__main__':
    rospy.init_node('object_detection')

    intrinsic_camera = load_intrinsics(eye_in_hand=True)
    object_finder = ObjectFinder(intrinsic_matrix=intrinsic_camera)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()
