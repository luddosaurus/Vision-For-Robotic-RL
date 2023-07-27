#! /usr/bin/env python3.8
import os

import rospkg
import rospy
import cv2
import numpy as np
from PIL import Image as pilimgage

from camera_calibration.params.calibration import calibration_path_d455, calibration_path_d435
from camera_calibration.utils.TypeConverter import TypeConverter
from camera_calibration.utils.TFPublish import TFPublish
from camera_calibration.utils.JSONHelper import JSONHelper

from utils.DaVinci import DaVinci
from utils.ColorObjectFinder import ColorObjectFinder
from utils.Const import Const

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

import tf2_ros
import actionlib
from my_robot_msgs.msg import MoveArmAction, MoveArmGoal, MoveArmResult, MoveArmFeedback

import matplotlib.pyplot as plt
import matplotlib.colors as col
import pandas as pd


# todo
# add finding system
# get image from topic aligned with depth
# get camera pose (world)
# get pose of object (camera) -> (world
# move arm to object


class ObjectFinder:

    def __init__(self, pose_estimate, camera_topic, intrinsic_matrix=None):
        # print(pose_estimate)
        self.pose_estimate = pose_estimate
        self.intrinsic_matrix = intrinsic_matrix
        self.cof = ColorObjectFinder()
        self.cv_bridge = CvBridge()
        self.depth_image = None

        self.center_x = None
        self.center_y = None
        self.center_z = None

        self.hovered_x = None
        self.hovered_y = None

        self.position = None

        self.camera_name = camera_topic

        # todo get camera pose in world frame
        self.window = 'ColorDetection'
        self.gui_created = False
        self.start_state = self.cof.get_state()
        self.current_image = None

        self.camera_subscriber = rospy.Subscriber(
            camera_topic + '/color/image_raw',
            Image, self.camera_color_callback)
        if self.pose_estimate:
            print('estimating pose')
            self.aligned_depth_subscriber = rospy.Subscriber(camera_topic + '/aligned_depth_to_color/image_raw', Image,
                                                             self.camera_depth_aligned_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)



        self.scale = 0.5
        self.roi_size = 9

        # self.center_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.center_broadcaster = tf2_ros.TransformBroadcaster()
        self.action_client = actionlib.SimpleActionClient('/pick_and_place', MoveArmAction)
        self.action_client.wait_for_server()

        self.world_to_cube_pickup = None
        self.world_to_cube_place = None

        self.cube_poses = {1: None, 2: None, 3: None, 4: None}
        self.total_height = 0

    def create_layout(self):
        cv2.namedWindow(self.window)
        # cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.createTrackbar("Hue", self.window,
                           self.start_state[Const.HUE], Const.HUE_MAX,
                           lambda value: self.cof.update_value(value, Const.HUE))
        cv2.createTrackbar("Saturation", self.window,
                           self.start_state[Const.SATURATION], Const.SAT_MAX,
                           lambda value: self.cof.update_value(value, Const.SATURATION))
        cv2.createTrackbar("Value", self.window,
                           self.start_state[Const.VALUE], Const.VAL_MAX,
                           lambda value: self.cof.update_value(value, Const.VALUE))

        cv2.createTrackbar("Hue Margin", self.window,
                           self.start_state[Const.HUE_MARGIN], Const.HUE_MAX,
                           lambda value: self.cof.update_value(value, Const.HUE_MARGIN))
        cv2.createTrackbar("Sat Margin", self.window,
                           self.start_state[Const.SATURATION_MARGIN], Const.SAT_MAX,
                           lambda value: self.cof.update_value(value, Const.SATURATION_MARGIN))
        cv2.createTrackbar("Val Margin", self.window,
                           self.start_state[Const.VALUE_MARGIN], Const.VAL_MAX,
                           lambda value: self.cof.update_value(value, Const.VALUE_MARGIN))

        cv2.createTrackbar("Noise", self.window,
                           self.start_state[Const.NOISE], Const.NOISE_MAX,
                           lambda value: self.cof.update_value(value, Const.NOISE))
        cv2.createTrackbar("Fill", self.window,
                           self.start_state[Const.FILL], Const.FILL_MAX,
                           lambda value: self.cof.update_value(value, Const.FILL))

        cv2.setMouseCallback(self.window, self.click)

    def update_scale(self, value):
        self.scale = value / 100

    def update_trackbars(self):
        current_state = self.cof.get_state()
        cv2.setTrackbarPos("Hue", self.window, current_state[Const.HUE])
        cv2.setTrackbarPos("Saturation", self.window, current_state[Const.SATURATION])
        cv2.setTrackbarPos("Value", self.window, current_state[Const.VALUE])
        cv2.setTrackbarPos("Hue Margin", self.window, current_state[Const.HUE_MARGIN])
        cv2.setTrackbarPos("Sat Margin", self.window, current_state[Const.SATURATION_MARGIN])
        cv2.setTrackbarPos("Val Margin", self.window, current_state[Const.VALUE_MARGIN])
        cv2.setTrackbarPos("Noise", self.window, current_state[Const.NOISE])
        cv2.setTrackbarPos("Fill", self.window, current_state[Const.FILL])

    def click(self, event, x, y, flags, param):
        self.hovered_x = x
        self.hovered_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            print("click!")
            self.cof.set_image_coordinate_color(
                image=self.current_image,
                x=x,
                y=y,
                scale=self.scale,
                roi_size=self.roi_size
            )
            self.update_trackbars()

    def camera_depth_aligned_callback(self, aligned_depth):
        # print(aligned_depth)
        aligned_input_depth = None
        try:
            aligned_input_depth = self.cv_bridge.imgmsg_to_cv2(
                aligned_depth, desired_encoding="passthrough")

        except CvBridgeError as e:
            print(e)
        self.depth_image = cv2.cvtColor(aligned_input_depth, cv2.COLOR_GRAY2BGR)

        # print(aligned_input_depth[a])
        # Find 3D point
        # cv2.imshow('test', aligned_input_depth)
        if self.center_x is not None and aligned_input_depth is not None and self.center_y is not None:
            depth_array = np.array(aligned_input_depth, dtype=np.float32)
            # print(depth_array.shape)
            if self.center_x <= depth_array.shape[1] and self.center_y <= depth_array.shape[0]:
                # print(self.center_x, depth_array.shape[1])
                depth = depth_array[self.center_y][self.center_x] / 1000

                # todo find depth of coordinate (x,y)
                position = self.cof.pixel_to_3d_coordinate((self.center_x, self.center_y), depth, self.intrinsic_matrix)
                # print(position)
                pose_info = f"x{position[0]:.2f} : y{position[1]:.2f}, z{position[2]:.2f}"

                self.center_z = position[2]
                self.position = position

                self.broadcast_point()

                # todo convert to world frame
                # todo publish

    def colorize_depth_image(self):
        # Normalize the depth values to a range suitable for color mapping
        normalized_depth = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply a color map to the normalized depth image
        self.depth_image = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

    def broadcast_point(self):
        # transform = TypeConverter.vectors_to_stamped_transform(translation=[self.x, self.y, self.z],
        #                                                        rotation=[0, 0, 0, 0, 0],
        #                                                        parent_frame='camera_estimateTSAI', child_frame='cube')
        # self.center_broadcaster.sendTransform
        # TFPublish.publish_transform(publisher=self.center_broadcaster, rotation=[0, 0, 0, 0, 0],
        #                             translation=[self.center_x, self.center_y, self.center_z], parent_name='camera_estimate0',
        #                             child_name='cube')
        TFPublish.publish_static_transform(publisher=self.center_broadcaster,
                                           parent_name=self.camera_name,
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

        hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)

        hues = hsv_image[:, :, :1]

        saturations = hsv_image[:, :, 1:2]
        values = hsv_image[:, :, 2:]

        # Mask
        mask_image = self.cof.get_hsv_mask(image=self.current_image)
        output = cv2.connectedComponentsWithStats(mask_image)

        res = cv2.bitwise_and(self.current_image, self.current_image, mask=mask_image)
        mask = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

        # Find center
        self.center_x, self.center_y = self.cof.find_mask_center(mask_image)
        if self.depth_image is not None:
            self.colorize_depth_image()
        pose_info = ""
        if self.center_x is not None:
            self.cof.draw_dot(res, self.center_x, self.center_y)
            self.cof.draw_dot(self.depth_image, self.center_x, self.center_y)
        display_image = self.current_image.copy()
        if self.hovered_x is not None:
            display_image = DaVinci.draw_roi_rectangle(image=display_image,
                                                       x=int(self.hovered_x / self.scale),
                                                       y=int(self.hovered_y / self.scale),
                                                       roi=self.roi_size)

        # Show Image
        mask_ = pilimgage.fromarray(mask_image)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            display_image = cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        stacked = np.hstack((display_image, res))

        info = "[0-9] states, [m]ove to, [q]uit"
        DaVinci.draw_text_box(
            image=stacked,
            text=info
        )

        slot_info = f"Color State [{self.cof.saved_state}]"
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
        # cv2.imshow('test', self.depth_image)
        # Input
        key = cv2.waitKey(1) & 0xFF
        key_str = chr(key)

        if key_str.isdigit() and 0 <= int(key_str) <= 9:
            key_number = int(key_str)
            # self.cof.current_state_index = key_number
            # self.update_trackbars()
            # print(f"Switching to state {key_number}")

            self.get_world_cube_transform(key_number)

        # elif key == ord('m'):
        #     world_to_cube = None
        #     while world_to_cube is None:
        #         try:
        #             world_to_cube = self.tf_buffer.lookup_transform('world', 'cube', rospy.Time())
        #         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #             print(f"No transform found between 'world' and 'cube'.")
        #     self.call_move_arm(world_to_cube)

        elif key == ord('u'):  # Pick up pose
            self.world_to_cube_pickup = None
            print('Waiting for transform world to cube...')
            while self.world_to_cube_pickup is None:
                try:
                    self.world_to_cube_pickup = self.tf_buffer.lookup_transform('world', 'cube', rospy.Time())
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    pass
            print(self.world_to_cube_pickup)

        elif key == ord('d'):  # Place pose
            self.world_to_cube_place = None
            print('Waiting for transform world to cube...')
            while self.world_to_cube_place is None:
                try:
                    self.world_to_cube_place = self.tf_buffer.lookup_transform('world', 'cube', rospy.Time())
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    pass
            print(self.world_to_cube_place)

        elif key == ord('m'):
            if self.world_to_cube_pickup is not None:
                self.call_move_arm(self.world_to_cube_pickup, self.world_to_cube_place)

        elif key == ord('s'):
            print(self.cube_poses)
            for key in self.cube_poses.keys():
                if key == 1:
                    continue
                self.call_move_arm(self.cube_poses[key], self.cube_poses[1])
                self.cube_poses[1].transform.translation.z += self.cube_poses[key].transform.translation.z

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
        elif key == ord('t'):
            # shape: (y, x, z)

            x_list = []
            y_list = []

            for x in range(hues.shape[0]):
                for y in range(hues.shape[1]):
                    x_list.append(x)
                    y_list.append(y)
            df_hue = pd.DataFrame(
                {'y': y_list, 'x': x_list, 'hue': hues.flatten()})
            # self.plot_3d(df_hue)
            df_h = pd.DataFrame({'hues': hues.flatten()})
            df_s = pd.DataFrame({'sat': saturations.flatten()})
            df_v = pd.DataFrame({'val': values.flatten()})

            fig, axes = plt.subplots(1, 3)
            df_h.hist('hues', ax=axes[0], bins=179)
            df_s.hist('sat', ax=axes[1], bins=255)
            df_v.hist('val', ax=axes[2], bins=255)
            plt.show()

    def plot_3d(self, dataframe):
        import matplotlib.cm as cm
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        x = dataframe['x']
        y = dataframe['y']
        z = dataframe['hue']
        # values for color
        c = [int(zv / 0.4) for zv in z]
        # discrete colormap with 3 colors
        cmap = col.ListedColormap(cm.tab10.colors[:len(np.unique(c))])
        axis.scatter(x, y, z, c=c, cmap=cmap)

        plt.show()

    def call_move_arm(self, pick_pose, place_pose):
        pick_pose_translation = pick_pose.transform.translation
        pick_translation = [pick_pose_translation.x, pick_pose_translation.y, pick_pose_translation.z]
        random_y = np.random.uniform(-0.3, 0.4)
        random_x = np.random.uniform(0.3, 0.45)
        place_translation = pick_translation[:1] + [random_y] + pick_translation[2:]

        move_arm_goal = MoveArmGoal()
        move_arm_goal.pickup_pose.position.x = pick_translation[0]
        move_arm_goal.pickup_pose.position.y = pick_translation[1]
        move_arm_goal.pickup_pose.position.z = pick_translation[2]
        if place_pose is None:
            move_arm_goal.place_pose.position.x = random_x
            move_arm_goal.place_pose.position.y = random_y
            move_arm_goal.place_pose.position.z = pick_translation[2]
        else:
            place_pose_translation = place_pose.transform.translation
            place_translation = [place_pose_translation.x, place_pose_translation.y, place_pose_translation.z]

            move_arm_goal.place_pose.position.x = place_translation[0]
            move_arm_goal.place_pose.position.y = place_translation[1]
            move_arm_goal.place_pose.position.z = place_translation[2] + pick_translation[2] + 0.02

        self.action_client.send_goal(move_arm_goal, feedback_cb=self.feedback_callback)

        self.action_client.wait_for_result()
        # print(self.action_client.get_state())

    def feedback_callback(self, m):
        print(m)

    def get_world_cube_transform(self, key):
        current_cube_transform = None
        print('Waiting for transform world to cube...')
        while current_cube_transform is None:
            try:
                current_cube_transform = self.tf_buffer.lookup_transform('world', 'cube', rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pass
        print(current_cube_transform)
        print(key)
        self.cube_poses[key] = current_cube_transform


def load_intrinsics(topics, intrinsic_names):
    print("ArUcoFinder launched with internal parameters:")
    camera_matrices = dict()
    for camera, topic in zip(intrinsic_names, topics):
        camera_intrinsics = JSONHelper.get_camera_intrinsics(camera)
        camera_matrix = np.array(camera_intrinsics['camera_matrix'])
        distortion = np.array(camera_intrinsics['distortion'])

        camera_matrices[topic] = camera_matrix
        print(camera_matrix, distortion)
    return camera_matrices


if __name__ == '__main__':
    rospy.init_node('object_detection')

    path = os.path.join(rospkg.RosPack().get_path('object_finder'), 'config/')
    config_file_name = rospy.get_param(param_name='object_detection/config')

    config_file_path = path + config_file_name

    parameters = JSONHelper.read_json(config_file_path)
    find_pose = parameters['find_pose']
    topics = parameters['camera_topic']
    intrinsic_names = parameters['camera_intrinsics']

    intrinsics = None
    if find_pose:
        intrinsics = load_intrinsics(topics=topics, intrinsic_names=intrinsic_names)

    # todo make sure topics ans intrinsics are aligned
    object_finder = ObjectFinder(
        pose_estimate=find_pose, camera_topic=topics[0], intrinsic_matrix=intrinsics[topics[0]]
    )

    # Update Freq
    rate = rospy.Rate(10)

    try:
        rospy.spin()
        rate.sleep()

    except KeyboardInterrupt:
        print('Shutting down.')

    cv2.destroyAllWindows()
