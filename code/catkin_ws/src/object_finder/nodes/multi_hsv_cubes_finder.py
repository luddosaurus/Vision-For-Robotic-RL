#! /usr/bin/env python3.8
import os

import rospkg
import rospy
import cv2
import numpy as np

from camera_calibration.params.calibration import calibration_path_d455, calibration_path_d435
from camera_calibration.utils.TypeConverter import TypeConverter
from camera_calibration.utils.TFPublish import TFPublish
from camera_calibration.utils.JSONHelper import JSONHelper

from utils.DaVinci import DaVinci
from utils.UI import UI
from utils.ColorObjectFinder import ColorObjectFinder

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

import tf2_ros
import actionlib
from my_robot_msgs.msg import MoveArmAction, MoveArmGoal, MoveArmResult, MoveArmFeedback
import geometry_msgs.msg as gm
import message_filters


class ObjectFinderController:

    def __init__(
            self,
            pose_estimation,
            camera_topics,
            camera_matrices=None
    ):

        self.camera_topics = camera_topics
        self.intrinsic_matrices = camera_matrices

        self.cof = ColorObjectFinder()
        self.ui = UI()
        self.cv_bridge = CvBridge()

        self.display_images = {}
        self.display_width = 640
        self.display_height = 480

        # Combined Image
        self.current_combined_image = None  # All images attached together

        self.roi_size = 9
        self.scale = 0.5

        self.current_image_dict = {topic: np.zeros((480, 640, 3)) for topic in camera_topics}
        self.current_segmented_image_dict = {topic: np.zeros((480, 640, 3)) for topic in camera_topics}

        self.selected_background_colors = []
        self.selected_block_colors = []

        # Camera COLOR Topics

        self.hand_subscriber = rospy.Subscriber(f'{camera_topics[0]}/color/image_raw', Image, callback=self.callback_hand)
        self.front_subscriber = rospy.Subscriber(f'{camera_topics[1]}/color/image_raw', Image, callback=self.callback_front)
        self.top_subscriber = rospy.Subscriber(f'{camera_topics[2]}/color/image_raw', Image, callback=self.callback_top)

        # Camera DEPTH Topics
        if pose_estimation:
            self.camera_depth_subscribers = {}
            for camera_topic in camera_topics:
                self.camera_depth_subscribers[camera_topic] = rospy.Subscriber(
                    name=f'{camera_topic}/aligned_depth_to_color/image_raw',
                    data_class=Image,
                    callback=self.camera_depth_callback,
                    callback_args=camera_topic
                )

        # Find Position
        self.segment_coordinates = {topic: dict() for topic in camera_topics}
        for topic in camera_topics:
            self.segment_coordinates[topic]["segment_centers_x"] = list()  # Pixel
            self.segment_coordinates[topic]["segment_centers_y"] = list()  # Pixel
            self.segment_coordinates[topic]["segment_centers_z"] = list()  # Depth
            self.segment_coordinates[topic]["positions"] = list()

        # Move Arm
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.center_broadcaster = tf2_ros.TransformBroadcaster()
        self.action_client = actionlib.SimpleActionClient('/pick_and_place', MoveArmAction)
        self.action_client.wait_for_server()

    def mouse_callback(self, event, x, y, flags, param):
        self.ui.update_mouse_hover(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cof.set_image_coordinate_color(
                image=self.current_combined_image,
                x=x, y=y,
                roi_size=self.roi_size,
                scale=self.scale
            )
            self.ui.update_trackbars(self.cof.get_state())

    # ----------------------------------------- Image Processing

    def camera_depth_callback(self, aligned_depth, topic_name):
        aligned_input_depth = None
        try:
            aligned_input_depth = self.cv_bridge.imgmsg_to_cv2(
                aligned_depth, desired_encoding="passthrough")

        except CvBridgeError as e:
            print(e)

        # Find 3D point
        segment_centers_x = self.segment_coordinates[topic_name]['segment_centers_x']
        segment_centers_y = self.segment_coordinates[topic_name]['segment_centers_y']
        self.segment_coordinates[topic_name]['segment_centers_z'].clear()
        self.segment_coordinates[topic_name]['positions'].clear()

        # for segment_center_x, segment_center_y in zip(segment_centers_x, segment_centers_y):
        for idx, (segment_center_x, segment_center_y) in enumerate(
                zip(segment_centers_x, segment_centers_y)):

            if segment_center_x is not None and aligned_input_depth is not None:
                depth_array = np.array(aligned_input_depth, dtype=np.float32)

                if segment_center_x <= depth_array.shape[1] and segment_center_y <= depth_array.shape[0]:
                    depth = depth_array[segment_center_y][segment_center_x] / 1000

                    position = self.cof.pixel_to_3d_coordinate(
                        pixel_coord=(segment_center_x, segment_center_y),
                        depth_value=depth,
                        camera_matrix=self.intrinsic_matrices[topic_name]
                    )

                    self.segment_coordinates[topic_name]['segment_centers_z'].append(position[2])
                    self.segment_coordinates[topic_name]['positions'].append(position)
                    self.broadcast_point(
                        point=position,
                        child_name=f'cube[{idx}]_from_{topic_name}',
                        parent_name=topic_name
                    )

    def callback_top(self, image):
        topic_name = 'cam_top'
        try:

            current_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            current_image = DaVinci.resize_and_crop_image(current_image, width=self.display_width,
                                                          height=self.display_height)
            self.current_image_dict[topic_name] = current_image
            self.segment(topic_name=topic_name, current_image=current_image)

        except CvBridgeError as e:
            print(e)

    def callback_front(self, image):
        topic_name = 'cam_front'
        try:

            current_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            current_image = DaVinci.resize_and_crop_image(current_image, width=self.display_width,
                                                          height=self.display_height)
            self.current_image_dict[topic_name] = current_image
            self.segment(topic_name=topic_name, current_image=current_image)

        except CvBridgeError as e:
            print(e)

    def segment(self, topic_name, current_image):
        # Remove background colors
        inv_mask = None
        if len(self.selected_background_colors) != 0:
            background_mask_image = self.cof.get_hsv_mask(
                image=current_image,
                color_list=self.selected_background_colors
            )
            inv_mask = cv2.bitwise_not(background_mask_image)

        # Add selected colors
        current_color_mask_image = self.cof.get_hsv_mask(
            image=current_image,
            color_list=self.selected_block_colors + [self.cof.get_current_state()])

        if inv_mask is not None:
            current_color_mask_image = cv2.bitwise_and(inv_mask, current_color_mask_image)

        # Segment image
        segmented_image = cv2.bitwise_and(
            src1=current_image,
            src2=current_image,
            mask=current_color_mask_image
        )

        # Update center coordinates
        self.segment_coordinates[topic_name]['segment_centers_x'] = list()
        self.segment_coordinates[topic_name]['segment_centers_y'] = list()
        segment_coordinates = self.cof.find_segment_coordinates(current_color_mask_image)

        # Draw Centers
        for coordinate in segment_coordinates:
            x, y = coordinate
            self.segment_coordinates[topic_name]['segment_centers_x'].append(x)
            self.segment_coordinates[topic_name]['segment_centers_y'].append(y)
            self.cof.draw_dot(segmented_image, x, y)

        # ---------------------------------

        # Save segment image
        self.current_segmented_image_dict[topic_name] = segmented_image

    def callback_hand(self, image):
        topic_name = 'cam_wrist'
        try:
            current_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            current_image = DaVinci.resize_and_crop_image(
                current_image,
                width=self.display_width,
                height=self.display_height
            )
            self.current_image_dict[topic_name] = current_image
            self.segment(topic_name=topic_name, current_image=current_image)

        except CvBridgeError as e:
            print(e)

        self.update_callback()

    def combine_images(self):
        # stack segmentation with camera horizontally
        display_images = list()
        for image, segment in zip(self.current_image_dict.values(), self.current_segmented_image_dict.values()):
            image_segmentation_combo = np.vstack((image, segment))
            display_images.append(image_segmentation_combo)

        # stack all the cameras horizontally
        images = tuple(display_images)
        stacked = np.hstack(images)

        self.current_combined_image = stacked
        self.ui.display_image = stacked.copy()

    def update_callback(self):
        if not self.ui.gui_created:
            self.ui.create_layout(
                start_state=self.cof.get_state(),
                update_value=self.cof.update_value,
                on_click=self.mouse_callback
            )

        self.combine_images()
        self.ui.update_ui(self.cof.current_state_index, self.scale, self.roi_size)
        self.read_input()

    def read_input(self):
        key = cv2.waitKey(1) & 0xFF
        key_str = chr(key)

        if key_str.isdigit() and 0 <= int(key_str) <= 9:
            key_number = int(key_str)
            self.cof.current_state_index = key_number
            self.ui.update_trackbars(self.cof.get_current_state())
            print(f"Switching to state {key_number}")

        elif key == ord('m'):
            self.move_arm()

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
        elif key == ord('r'):
            state = self.cof.get_current_state().copy()
            state[-2:] = [0, 0]  # set fill and noise to 0
            self.selected_background_colors.append(state)
        elif key == ord('c'):
            self.selected_background_colors = []
            self.selected_block_colors = []
        elif key == ord('a'):
            state = self.cof.get_current_state().copy()
            self.selected_block_colors.append(state)

    def broadcast_point(self, point, child_name, parent_name):
        TFPublish.publish_static_transform(publisher=self.center_broadcaster,
                                           parent_name=parent_name,
                                           child_name=child_name,
                                           rotation=[0., 0., 0., 1.],
                                           translation=point)

    def move_arm(self):
        world_to_cube = None
        transforms = []

        while world_to_cube is None:
            try:
                for camera_topic in self.camera_topics:
                    world_to_cube = self.tf_buffer.lookup_transform(
                        'world',
                        f'cube_from_{camera_topic}',
                        rospy.Time()
                    )
                    transforms.append(world_to_cube)

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print(f"No transform found between 'world' and 'cube'.")

        average_pose = self.compute_average_transform(transforms)
        self.call_move_arm(average_pose)

    @staticmethod
    def compute_average_transform(stamped_transforms):
        translation_sum = np.zeros(3)
        rotation_sum = np.zeros(4)

        for stamped_transform in stamped_transforms:
            transform = stamped_transform.transform
            translation_sum += np.array(
                [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z
                ]
            )

            rotation_sum += np.array(
                [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w
                ]
            )

        translation_avg = translation_sum / len(stamped_transforms)
        rotation_avg = rotation_sum / len(stamped_transforms)

        avg_transform = gm.TransformStamped()
        avg_transform.transform.translation = gm.Vector3(*translation_avg)
        avg_transform.transform.rotation = gm.Quaternion(*rotation_avg)

        return avg_transform

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
    topics = parameters['camera_topics']
    intrinsic_names = parameters['camera_intrinsics']

    intrinsics = None
    if find_pose:
        intrinsics = load_intrinsics(topics=topics, intrinsic_names=intrinsic_names)

    object_finder = ObjectFinderController(
        camera_matrices=intrinsics,
        camera_topics=topics,
        pose_estimation=find_pose
    )

    # Update Freq
    rate = rospy.Rate(10)

    try:
        rospy.spin()
        rate.sleep()

    except KeyboardInterrupt:
        print('Shutting down.')

    cv2.destroyAllWindows()
