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
import actionlib
from my_robot_msgs.msg import MoveArmAction, MoveArmGoal, MoveArmResult, MoveArmFeedback


class ObjectFinder:

    def __init__(
            self,
            camera_color_topics,
            camera_depth_topics=None,
            intrinsic_matrix=None
    ):
        self.pose_estimate = camera_depth_topics is not None
        self.intrinsic_matrix = intrinsic_matrix
        self.cof = ColorObjectFinder()
        self.cv_bridge = CvBridge()

        # UI
        self.window = 'ColorDetection'
        self.gui_created = False
        self.current_image = None

        # Camera COLOR Topics
        self.camera_subscribers = {}
        for camera_topic in camera_color_topics:
            self.camera_subscribers[camera_topic] = rospy.Subscriber(
                name=camera_topic,
                data_class=Image,
                callback=self.camera_color_callback
            )

        # Camera DEPTH Topics
        self.camera_subscribers = {}
        for camera_topic in camera_depth_topics:
            self.aligned_depth_subscriber = rospy.Subscriber(
                name=camera_topic,
                data_class=Image,
                callback=self.camera_depth_callback)

        # Find Position
        # todo separate this into list
        # Image pixel (x, y) depth (z)
        self.segment_coordinates = {}
        self.segment_center_x = None
        self.segment_center_y = None
        self.segment_center_z = None

        self.mouse_hover_x = None
        self.mouse_hover_y = None

        self.position = None

        self.scale = 0.5
        self.roi_size = 9

        # Move Arm
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        # self.center_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.center_broadcaster = tf2_ros.TransformBroadcaster()
        self.action_client = actionlib.SimpleActionClient('/pick_and_place', MoveArmAction)
        self.action_client.wait_for_server()

    def create_layout(self):
        start_state = self.cof.get_state()
        cv2.namedWindow(self.window)
        # cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.createTrackbar("Hue", self.window,
                           start_state[self.cof.HUE], self.cof.HUE_MAX,
                           lambda value: self.cof.update_value(value, self.cof.HUE))
        cv2.createTrackbar("Saturation", self.window,
                           start_state[self.cof.SATURATION], self.cof.SAT_MAX,
                           lambda value: self.cof.update_value(value, self.cof.SATURATION))
        cv2.createTrackbar("Value", self.window,
                           start_state[self.cof.VALUE], self.cof.VAL_MAX,
                           lambda value: self.cof.update_value(value, self.cof.VALUE))

        cv2.createTrackbar("Hue Margin", self.window,
                           start_state[self.cof.HUE_MARGIN], self.cof.HUE_MAX,
                           lambda value: self.cof.update_value(value, self.cof.HUE_MARGIN))
        cv2.createTrackbar("Sat Margin", self.window,
                           start_state[self.cof.SATURATION_MARGIN], self.cof.SAT_MAX,
                           lambda value: self.cof.update_value(value, self.cof.SATURATION_MARGIN))
        cv2.createTrackbar("Val Margin", self.window,
                           start_state[self.cof.VALUE_MARGIN], self.cof.VAL_MAX,
                           lambda value: self.cof.update_value(value, self.cof.VALUE_MARGIN))

        cv2.createTrackbar("Noise", self.window,
                           start_state[self.cof.NOISE], self.cof.NOISE_MAX,
                           lambda value: self.cof.update_value(value, self.cof.NOISE))
        cv2.createTrackbar("Fill", self.window,
                           start_state[self.cof.FILL], self.cof.FILL_MAX,
                           lambda value: self.cof.update_value(value, self.cof.FILL))

        cv2.setMouseCallback(self.window, self.click)

    def update_scale(self, value):
        self.scale = value / 100

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
        self.mouse_hover_x = x
        self.mouse_hover_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cof.set_image_coordinate_color(self.current_image, x, y, self.scale, self.roi_size)
            self.update_trackbars()

    def camera_depth_callback(self, aligned_depth):
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
        if self.segment_center_x is not None and aligned_input_depth is not None:
            depth_array = np.array(aligned_input_depth, dtype=np.float32)
            # print(depth_array.shape)
            if self.segment_center_x <= depth_array.shape[1] and self.segment_center_y <= depth_array.shape[0]:
                # print(self.center_x, depth_array.shape[1])
                depth = depth_array[self.segment_center_y][self.segment_center_x] / 1000

                position = self.cof.pixel_to_3d_coordinate((self.segment_center_x, self.segment_center_y), depth, self.intrinsic_matrix)
                # print(position)
                pose_info = f"x{position[0]:.2f} : y{position[1]:.2f}, z{position[2]:.2f}"

                self.segment_center_z = position[2]
                self.position = position

                self.broadcast_point()

    def broadcast_point(self):
        TFPublish.publish_static_transform(publisher=self.center_broadcaster,
                                           parent_name='eye_in_hand',
                                           child_name=f'cube',
                                           rotation=[0., 0., 0., 1.],
                                           translation=self.position)

    def camera_color_callback(self, input_image):
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(input_image, desired_encoding="bgr8")

        except CvBridgeError as e:
            print(e)

        if not self.gui_created:
            self.create_layout()
            self.gui_created = True
        # image = self.current_image

        # Mask
        mask_image = self.cof.get_hsv_mask(image=self.current_image)
        res = cv2.bitwise_and(self.current_image, self.current_image, mask=mask_image)
        mask = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

        # Find center
        self.segment_center_x, self.segment_center_y = self.cof.find_mask_center(mask_image)
        pose_info = ""
        if self.segment_center_x is not None:
            self.cof.draw_dot(res, self.segment_center_x, self.segment_center_y)

        if self.mouse_hover_x is not None:
            self.current_image = DaVinci.draw_roi_rectangle(image=self.current_image,
                                                            x=int(self.mouse_hover_x / self.scale),
                                                            y=int(self.mouse_hover_y / self.scale),
                                                            roi=self.roi_size)

        # Show Image
        stacked = np.hstack((self.current_image, res))

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

        cv2.imshow(self.window, cv2.resize(stacked, None, fx=self.scale, fy=self.scale))
        # cv2.imshow(self.window, stacked)

        # Input
        key = cv2.waitKey(1) & 0xFF
        key_str = chr(key)

        if key_str.isdigit() and 0 <= int(key_str) <= 9:
            key_number = int(key_str)
            self.cof.current_state_index = key_number
            self.update_trackbars()
            print(f"Switching to state {key_number}")

        elif key == ord('m'):
            world_to_cube = None
            while world_to_cube is None:
                try:
                    world_to_cube = self.tf_buffer.lookup_transform('world', 'cube', rospy.Time())
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    print(f"No transform found between 'world' and 'cube'.")
            self.call_move_arm(world_to_cube)

        elif key == ord('q'):
            rospy.signal_shutdown('Bye :)')
        elif key == ord('o'):
            self.scale -= 0.05
        elif key == ord('p'):
            self.scale += 0.05
        elif key == ord('k'):
            if self.roi_size > 1:
                self.roi_size -= 2
        elif key == ord('l'):
            self.roi_size += 2

    def call_move_arm(self, pick_pose):
        pick_pose_translation = pick_pose.transform.translation
        pick_translation = [pick_pose_translation.x, pick_pose_translation.y, pick_pose_translation.z]
        random_y = np.random.uniform(-0.3, 0.4)
        random_x = np.random.uniform(0.2, 0.45)
        place_translation = pick_translation[:1] + [random_y] + pick_translation[2:]

        move_arm_goal = MoveArmGoal()
        move_arm_goal.pickup_pose.position.x = pick_translation[0]
        move_arm_goal.pickup_pose.position.y = pick_translation[1]
        move_arm_goal.pickup_pose.position.z = pick_translation[2]

        move_arm_goal.place_pose.position.x = random_x
        move_arm_goal.place_pose.position.y = random_y
        move_arm_goal.place_pose.position.z = pick_translation[2]

        self.action_client.send_goal(move_arm_goal, feedback_cb=self.feedback_callback)
        # status = self.action_client.get_state()
        # self.action_client.wait_for_result()
        # print(self.action_client.get_state())

    def feedback_callback(self, m):
        print(m)


def load_intrinsics(eye_in_hand):
    camera_intrinsics = JSONHelper.get_camera_intrinsics('d435_480p_testing')
    camera_matrix = np.array(camera_intrinsics['camera_matrix'])
    distortion = np.array(camera_intrinsics['distortion'])

    print("ArUcoFinder launched with internal parameters:")
    print(camera_matrix, distortion)
    return camera_matrix


if __name__ == '__main__':
    rospy.init_node('object_detection')

    c_topics = rospy.get_param(param_name='object_detection/camera_color_topics')
    d_topics = rospy.get_param(param_name='object_detection/camera_depth_topics')

    intrinsic_camera = None
    if d_topics is not None:
        intrinsic_camera = load_intrinsics(eye_in_hand=True)
    object_finder = ObjectFinder(
        intrinsic_matrix=intrinsic_camera,
        camera_color_topics=c_topics,
        camera_depth_topics=d_topics
    )

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()
