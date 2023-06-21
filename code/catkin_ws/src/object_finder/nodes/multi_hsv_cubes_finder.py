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
            pose_estimation,
            camera_topics,
            camera_matrices=None
    ):
        self.camera_topics = camera_topics
        self.intrinsic_matrices = dict()
        for camera_topic, matrix in camera_topics, camera_matrices:
            self.intrinsic_matrices[camera_topic] = matrix

        self.cof = ColorObjectFinder()
        self.cv_bridge = CvBridge()

        # UI
        self.window = 'ColorDetection'
        self.gui_created = False
        self.current_images = {}

        # todo have everything in camera topics?
        for camera in camera_topics:
            self.current_images[camera] = None

        # Combined Image
        self.current_combined_image = None  # All images attached together
        self.mouse_hover_x = None
        self.mouse_hover_y = None
        self.scale = 0.5
        self.roi_size = 9

        # Camera COLOR Topics
        # todo pray that all topics are on the same format
        self.camera_color_subscribers = {}
        for camera_topic in camera_topics:
            callback = self.color_callback(camera_topic)
            self.camera_color_subscribers[camera_topic] = rospy.Subscriber(
                name=f'{camera_topic}/color/image_raw',
                data_class=Image,
                callback=callback
            )

        # Camera DEPTH Topics
        if pose_estimation:
            self.camera_depth_subscribers = {}
            for camera_topic in camera_topics:
                callback = self.depth_callback(camera_topic)
                self.camera_depth_subscribers[camera_topic] = rospy.Subscriber(
                    name=f'{camera_topic}/aligned_depth_to_color/image_raw',
                    data_class=Image,
                    callback=callback
                )

        # Find Position
        self.segment_coordinates = {}
        for topic in camera_topics:
            self.segment_coordinates[topic]["segment_center_x"] = None  # Pixel
            self.segment_coordinates[topic]["segment_center_y"] = None  # Pixel
            self.segment_coordinates[topic]["segment_center_z"] = None  # Depth
            self.segment_coordinates[topic]["position"] = None

        # Move Arm
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.center_broadcaster = tf2_ros.TransformBroadcaster()
        self.action_client = actionlib.SimpleActionClient('/pick_and_place', MoveArmAction)
        self.action_client.wait_for_server()

    # ----------------------------------------- UI

    def create_layout(self):
        start_state = self.cof.get_state()
        cv2.namedWindow(self.window)

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
            self.cof.set_image_coordinate_color(
                image=self.current_combined_image,
                x=x, y=y,
                roi_size=self.roi_size)

            self.update_trackbars()

    # ----------------------------------------- Image Processing

    def depth_callback(self, topic_name):
        def camera_depth_callback(aligned_depth):
            aligned_input_depth = None
            try:
                aligned_input_depth = self.cv_bridge.imgmsg_to_cv2(
                    aligned_depth, desired_encoding="passthrough")

            except CvBridgeError as e:
                print(e)

            # Find 3D point
            segment_center_x = self.segment_coordinates[topic_name]['segment_center_x']
            segment_center_y = self.segment_coordinates[topic_name]['segment_center_y']

            if segment_center_x is not None and aligned_input_depth is not None:
                depth_array = np.array(aligned_input_depth, dtype=np.float32)

                if segment_center_x <= depth_array.shape[1] and segment_center_y <= depth_array.shape[0]:

                    depth = depth_array[segment_center_y][segment_center_x] / 1000

                    position = self.cof.pixel_to_3d_coordinate(
                        pixel_coord=(segment_center_x, segment_center_y),
                        depth_value=depth,
                        camera_matrix=self.intrinsic_matrices[topic_name]
                    )

                    self.segment_coordinates[topic_name]['segment_center_z'] = position[2]
                    self.segment_coordinates[topic_name]['position'] = position

                    self.broadcast_point(
                        point=position,
                        parent_name=topic_name
                    )

        return camera_depth_callback

    def color_callback(self, topic_name):
        def camera_color_callback(color_image):
            current_image = None
            try:
                current_image = self.cv_bridge.imgmsg_to_cv2(color_image, desired_encoding="bgr8")

            except CvBridgeError as e:
                print(e)

            if not self.gui_created:
                self.create_layout()
                self.gui_created = True
            # image = self.current_image

            # Mask
            mask_image = self.cof.get_hsv_mask(image=current_image)
            segmented_image = cv2.bitwise_and(
                src1=self.current_images[topic_name],
                src2=self.current_images[topic_name],
                mask=mask_image
            )

            # Find center
            segment_center_x, segment_center_y = self.cof.find_mask_center(mask_image)
            self.segment_coordinates[topic_name]['segment_center_x'] = segment_center_x
            self.segment_coordinates[topic_name]['segment_center_y'] = segment_center_y

            # Show Image
            if segment_center_x is not None:
                self.cof.draw_dot(segmented_image, segment_center_x, segment_center_y)

            image_segmentation_combo = np.vstack((current_image, segmented_image))
            self.current_images[topic_name] = image_segmentation_combo

            images = list(self.current_images.values())
            stacked = np.hstack(images)
            self.current_combined_image = stacked

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

            if self.segment_coordinates[topic_name]['position']:
                position = self.segment_coordinates[topic_name]['position']*100
                pose_info = f"x{int(position[0])} : y{int(position[1])}, z{int(position[2])}"
                DaVinci.draw_text_box(
                    image=stacked,
                    text=pose_info,
                    position="top_right"
                )

            if self.mouse_hover_x is not None:
                self.current_combined_image = DaVinci.draw_roi_rectangle(
                    image=stacked,
                    x=int(self.mouse_hover_x / self.scale),
                    y=int(self.mouse_hover_y / self.scale),
                    roi=self.roi_size
                )

            cv2.imshow(
                self.window,
                cv2.resize(stacked, None, fx=self.scale, fy=self.scale)
            )

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
                # todo change to take an average of all?
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

        return camera_color_callback

    # ----------------------------------------- Move Arm
    def broadcast_point(self, point, parent_name):
        # todo could child have multiple parents?
        TFPublish.publish_static_transform(publisher=self.center_broadcaster,
                                           parent_name=parent_name,
                                           child_name=f'cube',
                                           rotation=[0., 0., 0., 1.],
                                           translation=point)

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

        self.action_client.send_goal(move_arm_goal, feedback_cb=self.arm_feedback)

    @staticmethod
    def arm_feedback(m):
        print(f'Arm Feedback : {m}')


def load_intrinsics(eye_in_hand):
    camera_matrices = []
    camera_intrinsics = JSONHelper.get_camera_intrinsics('d435_480p_testing')
    camera_matrix = np.array(camera_intrinsics['camera_matrix'])
    distortion = np.array(camera_intrinsics['distortion'])

    camera_matrices.append(camera_matrix)
    # todo add all the other intrinsics

    print("ArUcoFinder launched with internal parameters:")
    print(camera_matrix, distortion)
    return camera_matrices


if __name__ == '__main__':
    rospy.init_node('object_detection')

    find_pose = rospy.get_param(param_name='object_detection/find_pose')
    topics = rospy.get_param(param_name='object_detection/camera_topics')

    intrinsics = None
    if find_pose:
        intrinsics = load_intrinsics(eye_in_hand=True)

    # todo make sure topics ans intrinsics are aligned
    object_finder = ObjectFinder(
        camera_matrices=intrinsics,
        camera_topics=topics,
        pose_estimation=find_pose
    )

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down.')
    cv2.destroyAllWindows()
